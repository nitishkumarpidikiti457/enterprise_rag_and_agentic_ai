"""Evaluation harness.

Retrieval (always, no API key needed):
  hit@1, recall@5, MRR@5 and section precision@5 for dense / hybrid / hybrid+rerank.
Generation (when an LLM key is configured, or --providers given):
  answer correctness (LLM-as-judge), citation rate, refusal on unanswerable questions,
  p50/p95 latency and estimated cost per query, per provider.

    python eval/run_eval.py                      # retrieval only
    python eval/run_eval.py --generation         # + answers with every configured provider
    python eval/run_eval.py --generation --providers groq gemini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from app.config import get_settings
from app.ingestion.pipeline import ingest_path
from app.llm.cache import get_cache
from app.llm.providers import Message
from app.llm.router import get_router
from app.rag import answer_question
from app.retrieval.retriever import RetrievalConfig, retrieve

GOLDEN = Path(__file__).with_name("golden_set.jsonl")
RESULTS = Path(__file__).with_name("results.md")
UNANSWERABLE = [
    "What is the CEO's home address?",
    "How many parking spaces does the Tampa distribution center have?",
    "What is the stock price target for next year?",
]
CONFIGS = {
    "dense only": RetrievalConfig(hybrid=False, rerank=False),
    "hybrid (dense+BM25, RRF)": RetrievalConfig(hybrid=True, rerank=False),
    "hybrid + cross-encoder rerank": RetrievalConfig(hybrid=True, rerank=True),
}


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def retrieval_eval(golden: list[dict], k: int = 5) -> list[dict]:
    rows = []
    for name, base in CONFIGS.items():
        cfg = RetrievalConfig(top_k=20, top_n=k, hybrid=base.hybrid, rerank=base.rerank)
        hit1 = recall = rr = prec = 0.0
        lat = []
        for g in golden:
            t = time.perf_counter()
            hits = retrieve(g["question"], cfg)
            lat.append(time.perf_counter() - t)
            match = [h.metadata.get("source") == g["source"] and h.metadata.get("section") == g["section"]
                     for h in hits]
            same_doc = [h.metadata.get("source") == g["source"] for h in hits]
            hit1 += bool(match and match[0])
            recall += any(match)
            rr += next((1 / (i + 1) for i, m in enumerate(match) if m), 0.0)
            prec += sum(same_doc) / max(len(hits), 1)
        n = len(golden)
        rows.append({"config": name, "hit@1": hit1 / n, f"recall@{k}": recall / n, f"MRR@{k}": rr / n,
                     f"doc precision@{k}": prec / n,
                     "p50 ms": statistics.median(lat) * 1000})
    return rows


JUDGE = """You grade answers. Reference answer: "{ref}". Candidate answer: "{cand}".
Does the candidate state the same fact as the reference (extra detail is fine)? Reply with only YES or NO."""


async def judge(ref: str, cand: str) -> bool:
    ref_l, cand_l = ref.lower(), cand.lower()
    if ref_l in cand_l:
        return True
    try:
        r = await get_router().complete([Message("user", JUDGE.format(ref=ref, cand=cand))],
                                        task="classify", max_tokens=3, temperature=0.0)
        if r.provider != "extractive" and r.text.strip():
            return r.text.strip().upper().startswith("YES")
    except Exception:  # noqa: BLE001
        pass
    toks = [t for t in ref_l.replace(",", " ").split() if len(t) > 2]
    return bool(toks) and sum(t in cand_l for t in toks) / len(toks) >= 0.6


async def generation_eval(golden: list[dict], providers: list[str]) -> list[dict]:
    rows = []
    for prov in providers:
        get_cache().clear()
        correct = cited = refused = 0
        lat = []
        p_obj = get_router().providers[prov]
        for g in golden:
            t = time.perf_counter()
            res = await answer_question(g["question"], provider=prov)
            lat.append(time.perf_counter() - t)
            correct += await judge(g["answer"], res.answer)
            cited += bool(res.citations)
        for q in UNANSWERABLE:
            res = await answer_question(q, provider=prov)
            refused += "don't know" in res.answer.lower()
        n = len(golden)
        s = sorted(lat)
        rows.append({"provider": f"{prov} ({p_obj.model})", "answer accuracy": correct / n,
                     "citation rate": cited / n, f"refusals on {len(UNANSWERABLE)} unanswerable": refused,
                     "p50 s": statistics.median(lat), "p95 s": s[int(0.95 * (len(s) - 1))]})
    return rows


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", action="store_true")
    ap.add_argument("--providers", nargs="*")
    args = ap.parse_args()

    ingest_path("data/sample_docs")
    golden = load_golden()
    st = get_settings()
    out = [f"# Evaluation results\n\n{len(golden)} golden questions over {len({g['source'] for g in golden})} documents. "
           f"Embeddings: `{st.embedding_backend}` ({st.embedding_model}); reranker: `{st.reranker_backend}`.\n",
           "## Retrieval\n", to_markdown(retrieval_eval(golden))]
    if args.generation:
        provs = args.providers or [p for p in get_router().available() if p != "extractive"] or ["extractive"]
        out += ["\n## Generation\n", to_markdown(asyncio.run(generation_eval(golden, provs)))]
    text = "\n".join(out) + "\n"
    RESULTS.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
