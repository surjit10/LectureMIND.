# Benchmark analysis — `evaluation_report_20261114_171933.json`

## 95% bootstrap confidence intervals

| Metric | Mean | 95% CI | n |
|---|---:|---:|---:|
| post_rerank_hit_at_5 | 0.9070 | [0.8140, 0.9767] | 43 |
| post_rerank_mrr_at_5 | 0.8012 | [0.6907, 0.9031] | 43 |
| context_recall | 0.8430 | [0.7364, 0.9360] | 43 |
| hit_at_5 | 0.8837 | [0.7907, 0.9767] | 43 |
| mrr_at_5 | 0.8043 | [0.6919, 0.9031] | 43 |
| answer_f1 | 0.2559 | [0.2013, 0.3127] | 45 |
| keyword_recall | 0.7857 | [0.6815, 0.8778] | 45 |
| routing_accuracy | 0.9333 | [0.8444, 1.0000] | 45 |
| total_latency | 27.5120 | [25.7456, 28.9704] | 45 |

## Error analysis (answerable questions)

| Outcome | Questions | Share |
|---|---:|---:|
| ok | 36 | 83.7% |
| refused_with_evidence | 2 | 4.7% |
| lost_after_retrieval | 4 | 9.3% |
| retrieval_miss | 1 | 2.3% |

| Outcome | Type | Gold rank after rerank | Question |
|---|---|---:|---|
| lost_after_retrieval | conceptual | 13 | Why do the coefficients used for combining word embeddings vary across different words? |
| lost_after_retrieval | graph | 5 | How is Process Abstraction related to Process in the lecture graph? |
| lost_after_retrieval | graph | 14 | Which subsystem does Kernel connect to via USED_BY for memory handling? |
| lost_after_retrieval | visual | 14 | Which brand of microphone is visible in the podcast recording setup? |
| refused_with_evidence | visual | 1 | What device was used to run the live demonstration of the end‑to‑end audio model during th |
| refused_with_evidence | visual | 2 | What feature is highlighted in the educational slide as essential for capturing long‑range |
| retrieval_miss | factual | — | What are the three main factors that have enabled deep learning's recent success according |

`ok` means the evidence reached the model and it answered; answer correctness
itself is not judged here (no LLM judge has been run).
