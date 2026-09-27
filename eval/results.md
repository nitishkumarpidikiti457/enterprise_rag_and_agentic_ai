# Evaluation results

50 golden questions over 21 documents. Embeddings: `hash` (BAAI/bge-small-en-v1.5); reranker: `lexical`.

## Retrieval

| config | hit@1 | recall@5 | MRR@5 | doc precision@5 | p50 ms |
|---|---|---|---|---|---|
| dense only | 0.90 | 0.98 | 0.94 | 0.38 | 3.31 |
| hybrid (dense+BM25, RRF) | 0.94 | 1.00 | 0.97 | 0.48 | 3.33 |
| hybrid + cross-encoder rerank | 1.00 | 1.00 | 1.00 | 0.48 | 3.62 |

## Generation

| provider | answer accuracy | citation rate | refusals on 3 unanswerable | p50 s | p95 s |
|---|---|---|---|---|---|
| extractive (extractive-v1) | 1.00 | 1.00 | 0 | 0.00 | 0.00 |
