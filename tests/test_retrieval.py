import json
from pathlib import Path

from app.retrieval.retriever import RetrievalConfig, retrieve, rrf_fuse
from app.retrieval.vectorstore import SearchHit

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden_set.jsonl"


def test_rrf_fuse_rewards_agreement():
    a = [SearchHit("x", "", {}, 1), SearchHit("y", "", {}, 1)]
    b = [SearchHit("y", "", {}, 1), SearchHit("z", "", {}, 1)]
    fused = rrf_fuse([a, b], [0.5, 0.5])
    assert fused[0].chunk_id == "y"


def test_retrieves_correct_document(store):
    hits = retrieve("How quickly must a Class I recall be pulled from shelves?")
    assert hits[0].metadata["source"] == "product_recall_policy.pdf"
    assert hits[0].metadata["section"] == "Recall Classes"


def test_source_filter(store):
    hits = retrieve("refund", RetrievalConfig(source="return_and_refund_policy.docx"))
    assert hits and all(h.metadata["source"] == "return_and_refund_policy.docx" for h in hits)


def test_golden_set_recall_at_5(store):
    golden = [json.loads(line) for line in GOLDEN.read_text().splitlines()]
    found = sum(any(h.metadata["source"] == g["source"] for h in retrieve(g["question"])) for g in golden)
    assert found / len(golden) >= 0.9
