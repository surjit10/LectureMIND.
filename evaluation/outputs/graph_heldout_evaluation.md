# LectureMIND — Held-Out GraphRAG Evaluation Report

**Dataset:** `evaluation/datasets/cs162_lecture1_graph_qa_heldout.json`  
**Run timestamp:** 2026-09-25T17:05:00+05:30  
**Sample size:** 30 questions (8 1-hop, 10 2-hop, 7 3-hop, 5 negative)  
**Question / label provenance:** see `evaluation/outputs/graph_heldout_creation_metadata.json` and `evaluation/outputs/graph_heldout_manual_validation.json`.  

---

## 1. Summary

| Metric | Result | How it is measured |
|---|---:|---|
| Structural path recovery (answerable) | 13/25 (52.0%) | Every gold relation of the question appears among the retrieved Neo4j edges (entity-name match, direction ignored) |
| Hybrid chunk Hit@5 (answerable) | 84.0% | Graph + BM25 fused by RRF. Versus BM25 alone: Hit@5 +20.0 pts, Hit@1 -24.0 pts, MRR -9.8 pts |
| Negative refusals | 5/5 (100.0%) | Generated answer contains a refusal phrase |
| False-positive path rate (negatives) | 1/5 (20.0%) | Graph returned an edge directly linking two trap entities |
| Lenient answer match (overall) | 25/30 (83.3%) | Answerable: not a refusal AND (gold-entity keyword recall ≥ 0.5 OR token F1 ≥ 0.25). 9 of 20 answerable matches passed on keyword recall alone (F1 < 0.25). This is a lexical proxy, not a correctness judgment. |

---

## 2. Hop-by-Hop Breakdown

| Category | N | Path recovery | Hybrid Hit@5 | Lenient answer match | Mean token F1 |
|---|---:|---:|---:|---:|---:|
| 1-hop | 8 | 4/8 (50.0%) | 5/8 (62.5%) | 7/8 (87.5%) | 0.234 |
| 2-hop | 10 | 7/10 (70.0%) | 10/10 (100.0%) | 6/10 (60.0%) | 0.165 |
| 3-hop | 7 | 2/7 (28.6%) | 6/7 (85.7%) | 7/7 (100.0%) | 0.232 |
| Negative | 5 | 1 false-positive path(s) | n/a | 5/5 (100.0%) refused | n/a |

---

## 3. Evidence Chunk Retrieval (answerable questions)

| Mode | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR (full list) |
|---|---:|---:|---:|---:|---:|
| BM25 lexical | 0.280 | 0.440 | 0.640 | 0.353 | 0.438 |
| Graph-only | 0.000 | 0.000 | 0.080 | 0.033 | 0.142 |
| Hybrid (Graph + BM25, RRF) | 0.040 | 0.560 | 0.840 | 0.500 | 0.340 |

---

## 4. Negative Questions

- **GQ26:** What direct prerequisite relationship exists between the Global Data Plane and CPU Scheduling in the lecture? — graph linked trap entities: no; answer refused: yes
- **GQ27:** How is Bell's Law a prerequisite for the implementation of semaphores in the kernel? — graph linked trap entities: no; answer refused: yes
- **GQ28:** What 2-hop path connects ARPANET to the translation lookaside buffer through the file system? — graph linked trap entities: no; answer refused: yes
- **GQ29:** How does the instruction set architecture directly explain the grading policy of the course? — graph linked trap entities: no; answer refused: yes
- **GQ30:** Trace how Git version control serves as a prerequisite for hardware cache coherence protocols in multi-core processors. — graph linked trap entities: yes; answer refused: yes

---

## 5. Scope and Caveats

1. **Not the production pipeline.** Chunk retrieval here uses `SimpleBM25` + `SimpleGraphRetriever` fused by RRF, and the prompt context is the top-3 chunks plus up to 5 graph edges. Dense retrieval, cross-encoder reranking and `ContextBuilder` are not exercised, so these numbers do not measure the deployed query workflow.
2. **Small sample.** 8/10/7 answerable questions per hop and 5 negatives; one question moves the smallest per-hop rate by 14 points.
3. **Lexical answer scoring.** "Lenient answer match" is keyword/F1 based; gold entity names usually appear in the question itself. Use a calibrated judge or human review for answer correctness.
4. **Cached generations.** Answers are reused from `evaluation/outputs/graph_heldout_checkpoint.json` when present; delete it to regenerate answers after pipeline changes.