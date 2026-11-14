# Retrieval comparison — commit 891648c vs. current correctness fixes (2026-11-14)

No LLM calls. Both code versions ran the real planner → graph retrieval (graph routes)
→ dense retrieval → BM25 + RRF → `bge-reranker-base` cross-encoder → `ContextBuilder`
against the same live Qdrant/Neo4j data, the same `bge-large-en-v1.5` embedder and the
same reranker weights (`local_runtime/models/global_reranker`, FP32, CPU).

- Dataset: `evaluation/datasets/multilecture_smoke_45.json` (45 questions, 43 answerable,
  3 lectures; labels auto-generated, **not human-reviewed**).
- Plus 3 summary prompts per lecture ("Summarize the lecture", "What are the key
  takeaways?", "Give me an overview of today's lecture") to measure lecture-wide coverage.
- Old code: `git worktree` of `891648c`. New code: working tree with the correctness fixes.
- Raw per-query records: `old_code.jsonl`, `new_code.jsonl`; aggregates: `comparison.json`.
- Reproduce: `scripts/compare_retrieval_run.py` + `scripts/compare_retrieval_analyze.py`.

Consistency check: the old-code pre-rerank Hit@5 (0.8605), MRR@5 (0.7333) and context
recall (0.7674) reproduce the stored Groq report `smoke_groq_20261010_fixed` exactly.

## Benchmark questions (43 answerable)

| Metric | Old code | New code |
|---|---:|---:|
| Pre-rerank Hit@5 (RRF candidate order) | 0.8605 | 0.8605 |
| Pre-rerank MRR@5 | 0.7333 | 0.7333 |
| Post-rerank Hit@1 | 0.6977 | 0.6977 |
| Post-rerank Hit@5 | 0.8837 | 0.8837 |
| Post-rerank Recall@5 | 0.8721 | 0.8721 |
| Post-rerank MRR@5 | 0.7593 | 0.7593 |
| Post-rerank nDCG@5 | 0.7854 | 0.7854 |
| Candidate-pool recall (gold chunk anywhere in the reranked pool) | 0.9302 | 0.9302 |
| Context recall (gold chunk placed in the LLM prompt) | 0.7674 | 0.7674 |
| Routing accuracy (45 questions) | 0.8889 | 0.9333 |
| Graph-routed questions | 10 | 8 |
| Mean `[Graph]` lines in context, graph-routed questions | 24.6 | 41.8 |
| Mean retrieval latency | 0.167 s | 0.185 s |
| Mean rerank + context latency | 4.93 s | 5.37 s |

Per-question post-rerank MRR@5: 0 better, 0 worse, 43 identical. The two routing
changes are the "fully connected layer" questions (graph_only → vector_only, now correct).

Reading the table:
- Chunk-level retrieval is unchanged. This set has no lecture-wide, grading/syllabus or
  plural-"prerequisites" questions, and graph results contribute relation lines, not chunks.
- The graph fix shows up as more relation lines in context (incoming edges are now included).
- The rerank-latency difference is run-to-run noise: questions that take the identical
  `vector_only` code path in both versions also moved 4.95 s → 5.35 s. The real cost of the
  two-direction traversal is in graph-routed retrieval: 0.30 s → 0.40 s.
- The reranker adds +0.023 Hit@5 and +0.026 MRR@5 over the RRF order (about one question).
- 93% of gold chunks enter the candidate pool but only 77% reach the prompt: the main loss is
  between reranking and context selection, not first-stage retrieval.

## Lecture-wide summaries (9 prompts)

| Metric | Old code | New code |
|---|---:|---:|
| Mean timeline span covered by the context | 0.807 | 0.872 |
| Worst prompt's timeline span | 0.464 | 0.863 |
| Mean lecture deciles covered (of 10) | 5.33 | 7.67 |
| Worst prompt's deciles covered | 4 | 7 |
| Timeline blocks reaching the prompt (mean) | 5.7 | 8.0 |

Two fixes together produce this: lecture-wide queries no longer pass through BM25/RRF
truncation, and `ContextBuilder` gives each sampled timeline block an equal share of the
budget. With only the first fix the mean span was 0.622 — whole blocks were filled
chronologically, so the end of the lecture was cut by the 6,000-character budget.
