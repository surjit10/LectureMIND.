# LectureMind Runtime and Evaluation Forensic Audit

Audit date: 2026-10-10  
Scope: current source, configuration, local artifacts, saved evaluation outputs, dependency state, and safe local checks.  
Method: read-only inspection. No application code, database, model, dataset, API key, or environment setting was changed by this audit.

## A. Executive verdict

LectureMind is a functioning single-pass RAG application when its external services and provider are available. The user-run provider test returned success for Groq, and a real /query returned an answer with lecture-scoped sources and about 1.12 seconds total latency. The local registry currently contains four registered lectures; scripts/verify_registry.py --json returned status OK with four verified, zero problems, and reconciliation dry-run returned zero actions.

The current checkout is not equivalent to the historical benchmark run:

- At the start of the audit, `.env` set `ENABLE_AUTO_MODEL_RECOVERY=false` and the reranker directory was absent. During the follow-up remediation, `BAAI/bge-reranker-base` was downloaded into `local_runtime/models/global_reranker/` and the setting was changed to true.
- Before remediation, rerank_service.py intentionally preserved original order and copied the input score into rerank_score. After remediation, a clean live server loaded the model and produced changed cross-encoder scores.
- The current virtual environment is inconsistent. sentence-transformers import fails during test collection; metadata reports sentence-transformers 6.1.0 with transformers 4.57.6, while local_requirements.txt pins sentence-transformers==3.3.1. There are two Torch dist-info directories; import torch resolves to 2.14.1+cu130 while pip metadata also reports 2.3.1+cpu.
- Historical retrieval aggregates are reproducible from saved JSON, but three narrative values are wrong, dense-only ablation has no execution artifact, and the benchmark covers one lecture and 50 curated questions.
- Perfect citation/provenance and circular graph scores do not establish factual correctness. The held-out graph artifact is less perfect: structural path recovery 13/25 = 52%, hybrid downstream Hit@5 21/25 = 84%, and negative refusal 5/5.

Verdict: the demonstrated online RAG path works, but current reranker status and environment provenance are not trustworthy enough for an unqualified quality claim or production readiness.

| Area | Verdict | Evidence |
|---|---|---|
| API to answer path | Implemented single pass | Source trace; user live response |
| Online generation | Working | User provider test and /query |
| Dense/BM25/graph isolation | Guards present | Source and targeted tests; live DB unavailable here |
| Cross-encoder reranker | Active in clean verification server; old port-8000 process must be restarted | Local model loaded, ranking changed, `reranker_latency=3.5794`; persistent server restart still required |
| GraphRAG | Implemented, quality mixed | Held-out artifact plus source |
| Historical benchmark | Reproducible but limited | Raw JSON and verification artifacts |
| Current tests | Clean environment green for non-serving code | 576 targeted tests passed; service-dependent serving tests remain unverified |
| Production readiness | Not demonstrated | No current load/RAGAS report or live DB run here |

## B. Actual runtime architecture

### HTTP and lifecycle

serving/fastapi/routes/query.py:205-261 handles POST /query. Supplying lecture_id calls activate_lecture(): it requires a READY registry entry, clears the previous workflow, creates fresh Qdrant and Neo4j clients, loads the package, and persists the active lecture (query.py:78-120). Without a workflow the endpoint returns 503. Pipeline exceptions are converted to a QueryResponse whose answer contains an operational error string rather than returning a structured 5xx (query.py:254-261).

The developer trace is a rounded projection of workflow state, not a second retrieval pass (query.py:129-202). frontend/components/chat/PipelineTrace.tsx:91-113 displays vector and rerank numbers without labeling their semantic type.

### Planner and route behavior

agent/langgraph/workflow.py:50-142 executes planner, conditional graph retrieval, vector retrieval, reranker/context, and answer generation in one pass. Graph retrieval runs for graph_only and graph_and_vector. Vector retrieval runs for all three routes, including graph_only (workflow.py:96-118), because graph rows carry relationships but normally lack lecture passage text.

agent/dspy/planner.py:164-303 is a deterministic regex/keyword classifier unless a DSPy module is injected. Routes are vector_only, graph_only, and graph_and_vector. Visual is a need_visual flag, not a separate route. Planner-only checks in this audit produced:

| Example | Route | Intent | Lecture-wide | Visual |
|---|---|---|---|---|
| What is a process? | vector_only | factual_qa | no | no |
| Explain how virtual memory works | vector_only | factual_qa | no | no |
| What is the main topic of this lecture? | vector_only | summary | yes | no |
| How is Process Abstraction connected to Threads through Process? | graph_only | factual_qa | no | no |
| What does the slide show about the architecture diagram? | vector_only | factual_qa | no | yes |
| What is the capital of France? | vector_only | factual_qa | no | no |

The route choices are plausible, but the intent label for ordinary conceptual wording is weak, and there is no dedicated unanswerable/adversarial route.

### Retrieval

Dense retrieval is retrieval/vector_retriever/qdrant_retriever.py:78-178. It embeds with normalized BAAI/bge-large-en-v1.5, requires lecture_id, filters Qdrant by lecture_id, fetches slack candidates, removes transcript-only filler, and returns Qdrant similarity. Missing lecture_id raises instead of permitting a cross-lecture search (:135-145).

Lecture-wide retrieval is qdrant_retriever.py:181-278. It bypasses semantic search, scrolls a lecture-filtered collection, sorts by timestamp, samples clustered timeline windows, deduplicates, and emits score=1.0 (:244-275). That value means “selected by representative sampling,” not cosine relevance.

BM25 is retrieval/hybrid/bm25_retriever.py:46-124 and is cached by lecture_id (:171-272). RRF uses 1/(60+rank), deduplicates by chunk ID, and orders candidates before reranking (:277-344). BM25, dense, and RRF scores are different scales; only rank fusion is meaningful across legs.

Graph retrieval is retrieval/graph_retriever/neo4j_retriever.py:256-343. It requires lecture_id, performs exact, substring, and fuzzy entity matching. _build_cypher scopes both start and related nodes by lecture_id and traverses closed relations for up to three hops (:197-253). Loader relationships carry lecture_id (local/loaders/neo4j_loader.py:120-137).

### Reranking, context, answer, and sources

retrieval/reranker/rerank_service.py:86-225 merges by chunk_id, optionally performs BM25/RRF, builds passage text with visual content only when requested, and calls the active RerankerService. When no service exists and recovery is disabled, it logs a warning, preserves order, and assigns rerank_score = r.get("score", 0.0) (:187-218).

answer_generator.py:45-78 builds sources from context-tagged reranked entries and copies chunk ID, timestamp, and segment ID. Empty context returns “Insufficient evidence found in lecture.” Otherwise the LLM receives a prompt requiring claims to come from context (answer_generator.py:137-157, 184-235). This is provenance hygiene, not an entailment verifier.

ProviderRegistry.get_active_backend() selects the configured online provider when inference_mode is online and the active ID resolves; otherwise it falls back to Ollama (local/llm/provider_registry.py:264-300). The current local registry contains online Groq openai/gpt-oss-20b with a configured key; the key was not printed.

## C. Reranker investigation

| Question | Finding | Evidence |
|---|---|---|
| Model ID configured? | Yes, BAAI/bge-reranker-base | .env:24-30, config.py:150-170 |
| Auto recovery enabled? | Yes in the follow-up clean runtime | `.env`, verified smoke runner |
| Model in checkout? | Yes, `BAAI/bge-reranker-base` | `local_runtime/models/global_reranker/` |
| Dependency import works? | Yes in `.venv_clean` | 576 non-serving tests passed |
| Loaded in follow-up benchmark? | Yes | `reranker_latency` present in 45-sample report |
| Executed by follow-up benchmark? | Yes | Cross-encoder scores and ranking quality recorded |
| Fallback behavior | Still available if loading fails | `rerank_service.py:187-218` |
| Does fallback destroy all retrieval? | No; it removes learned ordering | Workflow/source inspection |
| Is setting recovery=true alone sufficient? | No | Requires compatible dependencies, model, disk, and restart |

The user’s live lecture-summary trace showed all vector and rerank scores equal to 1.0. This is explained by two code paths: lecture-wide sampling hardcodes dense score=1.0, and inactive-reranker fallback copies it. This is expected sampling behavior combined with a misleading score contract. It is not proof that a cross-encoder assigned equal relevance.

### Score semantics

| Score field | Producer | Calculation | Meaning | Relevance trust | Evidence |
|---|---|---|---|---|---|
| Qdrant score, ordinary query | Qdrant | Similarity of normalized query/vector | Dense similarity within a filtered run | Useful for ordering, not probability | qdrant_retriever.py:121-168 |
| Qdrant score, lecture-wide | Timeline sampler | Constant 1.0 | Representative-window selection | No | qdrant_retriever.py:244-275 |
| BM25 score | Local BM25 | Okapi BM25 k1=1.5, b=.75 | Lexical match | Only within BM25 | bm25_retriever.py:22-159 |
| RRF score | Fusion | Reciprocal rank sum with k=60 | Rank agreement | Ordering only | bm25_retriever.py:277-338 |
| rerank_score, active | CrossEncoder | model.predict(query, passage) | Model output | Useful with model/version provenance | local/services/reranker_service.py:30-55 |
| rerank_score, inactive | Fallback | Copy of input score | Original-order/sampling placeholder | No | rerank_service.py:215-218 |
| Graph fuzzy score | Entity matcher | Stem, token, sequence similarity | Entity-name matching | Not passage relevance | neo4j_retriever.py:61-152 |

The frontend developer panel displays both placeholder and genuine values identically. RRF score is not exposed in the trace, which limits fusion debugging.

## D. Retrieval and GraphRAG audit

Confirmed:

- Qdrant rejects missing lecture IDs and filters by lecture ID.
- Neo4j filters both ends of a traversed path by lecture ID.
- Neo4j loader writes lecture_id on relationships.
- RRF deduplicates chunks.
- Filler filtering preserves OCR/visual evidence.
- Registry hash verification currently passes for all four registered lectures.

Limitations:

- Graph entity matching is heuristic and can match common terms through substring/fuzzy logic.
- Graph rows generally lack transcript evidence; extracted relation correctness is therefore not guaranteed by traversal correctness.
- graph_only still executes vectors; this is defensible for grounded answers but route naming can mislead evaluation.
- Lecture-wide sampling is coverage-oriented rather than query-relevant and can miss narrow facts.
- No live Qdrant/Neo4j service was reachable from this audit shell, so current cross-lecture results and paths were not freshly exercised.

## E. Independent metric verification

### Historical 50-question benchmark

evaluation/datasets/cs162_lecture1_qa_50.json has 50 questions from one lecture: 23 factual, 17 conceptual, 5 visual, 4 definition, 1 summary; difficulty 22 easy, 26 medium, 2 hard. All expected routes are vector_only; there are no unanswerable/adversarial examples.

Raw source: evaluation/outputs/evaluation_report_20260811_105621.json. Verification: evaluation/outputs/question_level_metric_verification.json and reproducibility_matrix.json. Historical average cross-encoder score is varied (mean 0.08597, min 0.00054, max 0.49230), so it is a different run from the current disabled-reranker trace.

| Metric | Reported | Verified | N | Assessment |
|---|---:|---:|---:|---|
| Pre Hit@1 | .620 | .660 (33/50) | 50 | Historical narrative typo |
| Pre Hit@3 | .940 | .880 (44/50) | 50 | Historical narrative typo |
| Hybrid Hit@5 | .980 | .980 (49/50) | 50 | Reproducible on one curated lecture |
| Recall@5 | .862 | .862 | 50 | Binary expected-chunk recall |
| Strict MRR@5 | .7853 | .7853 | 50 | Reciprocal rank capped at 5 |
| Unbounded MRR | .788 | .7882 | 50 | Diagnostic, different definition |
| nDCG@5 | .767 | .7674 | 50 | Binary relevance, not human graded |
| Post Hit@1 | .840 | .860 (43/50) | 50 | Historical narrative under-reported |
| Post Hit@3 | .960 | .960 | 50 | Reproducible |
| Post Hit@5 | .980 | .980 | 50 | Reproducible |
| Post MRR/ranking quality | .918 | .9183 | 50 | First expected chunk rank |
| BM25 Hit@5 | .380 | .380 (19/50) | 50 | Reproducible lexical baseline |
| Dense-only ablation | .940 | Not verifiable | 50 | No standalone artifact |
| Answer token F1 | .459 | .4594 | 50 | Lexical overlap, paraphrase-sensitive |
| Keyword recall | .545 | .5453 | 50 | Label/substring dependent |
| Lexical Jaccard | .328 | .3285 | 50 | Not semantic faithfulness |

Implementations are in evaluation/metrics/retrieval_metrics.py and answer_metrics.py. Retrieval relevance is binary membership in expected_chunk_ids. Precision@k uses constant denominator k. Hit/MRR/nDCG contain vacuous empty-gold handling, though dataset validation is intended to reject empty gold sets.

### Why perfect values occur

- citation_completeness=1.0 checks that cited IDs exist in retrieved/context IDs (evaluation/metrics/citation_metrics.py:21-76). Sources are built from those entries (answer_generator.py:45-78), so the result is mainly structural and does not test claim entailment.
- chunk_coverage=1.0 divides cited IDs by reranked IDs (citation_metrics.py:82-99); it is not citation precision.
- Routing is 12/12 on a dataset whose wording mirrors planner regexes: 5 graph, 5 vector, 2 hybrid. This is a unit/integration test, not semantic generalization.
- The old graph report’s 20/20 paths were created from known graph paths. It tests Cypher/path integration, not independent GraphRAG QA.
- Nearest-rank p95 for binary metrics becomes 1.0 when at least 95% succeed; it does not mean every request succeeds.
- Reranker training MRR .996 is a 1-positive/1-negative objective from training-package triplets, not end-to-end retrieval.

### Held-out GraphRAG

evaluation/outputs/graph_heldout_evaluation.json contains 30 questions: 25 answerable and 5 negative. It reports structural recovery 13/25=.52 (1-hop 4/8, 2-hop 7/10, 3-hop 2/7), hybrid downstream Hit@5=.84 (21/25), answerable accuracy=.80 (20/25), and five correct negative refusals. It also records one false-positive path and false_positive_rate=.2, so negative refusal must be read with that caveat.

### Citation, factuality, RAGAS, operations

No claim-level entailment, unsupported-claim rate, citation precision/recall, or timestamp-error metric is implemented. evaluation/ragas/eval_ragas.py requires optional ragas/datasets and its CLI supplies one sample with empty ground truth. It also expects source text, while the /query response exposes IDs/timestamps/segment IDs. No evaluation/reports/ragas_report.json exists.

The historical report has sequential 50-query stage latency (mean total 20.5 seconds, p95 22.8 seconds). The load-test implementation now uses actual ThreadPoolExecutor workers and reports p50/p95/p99, throughput and errors, but evaluation/reports/load_test_report.csv does not exist. Current throughput, memory, saturation, and production error rates are unverified.

## F. Benchmark reliability

| Dataset | Classification | Reason |
|---|---|---|
| cs162_lecture1_qa_50.json | Curated development benchmark | One lecture, 50 questions, no adversarial/unanswerable set |
| cs162_lecture1_routing_12.json | Unit/integration test | Small and pattern-matched |
| cs162_lecture1_graph_qa.json | Targeted graph integration test | Known-path construction |
| cs162_lecture1_graph_qa_heldout.json | Held-out development benchmark | Better independence, still one lecture/N=30 |
| transformer_kg_qa_50.json | Dataset artifact | No current full end-to-end report located |
| KG gold files | Pending-human-verification audit labels | METRICS_MATRIX.md:147-149 says LLM-assisted and Transformer-only |

The repository has a 70/15/15 splitter in evaluation/retrieval/eval_retrieval.py:29-36, but the saved 50-question report is not documented as an independent split. The reranker metric explicitly lacks held-out generalization. There is no production-scale multi-lecture human-blind benchmark.

## G. Previously implemented corrections

| Requested correction | Current evidence | Audit result |
|---|---|---|
| Lecture-scoped Qdrant point IDs/filters | qdrant_retriever.py:135-145 requires lecture_id and applies a payload filter; isolation tests exist | Source/test verified; live Qdrant unavailable |
| A10 prerequisite inference | cloud extraction and local prerequisite_enricher modules exist; prerequisite schema/loader tests passed in the targeted run | Implementation present; no fresh graph execution |
| prerequisites.json validation/export | manifest_validator.py validates the artifact and prerequisite count; prerequisite tests passed | Verified in source/tests |
| DAG-cycle detection | prerequisite/graph audit modules and tests are present; saved audit reports show zero prerequisite cycles | Verified artifact/source; not regenerated here |
| Registry hash verification | verify_registry returned four verified, zero problems; reconcile dry-run had zero actions | Live filesystem verified |
| ZIP versus extracted-package handling | verify_registry requires package_path to be a directory; reconcile never deletes ZIPs or DB data | Source verified |
| Safe reconciliation | reconcile_registry.py default is dry-run and reports destructive_actions=0 | Verified by execution |
| Reranker train/dev leakage prevention | METRICS_MATRIX.md documents the training metric as same-package pairwise convergence without a held-out split; no independent leakage audit artifact was found for the current model | Not proven; benchmark claim must remain qualified |
| RAGAS validation | eval_ragas.py refuses optional-missing dependencies and no longer writes placeholder zeros | Harness guard verified; valid RAGAS result absent |
| Realistic load-test concurrency | load_test.py uses ThreadPoolExecutor(max_workers=level), reports p50/p95/p99 and throughput | Source verified; no report executed |
| MLOps run manifests | mlops/run_manifest.py records git state, Python/platform, dependencies, and parameters | Source and current smoke-report provenance verified |
| CI benchmark checks | .github/workflows/ci.yml compiles code and validates benchmark structure, then runs unit tests; it explicitly does not fake live retrieval | Source verified; current local dependency drift would fail reranker collection |
| Docker routing | local compose routes backend dependencies by localhost defaults; production compose routes backend to service names | Source verified; Docker unavailable here |
| Production security settings | production compose requires NEO4J_PASSWORD, but local compose uses NEO4J_AUTH=none and production publishes DB ports without an API auth layer | Partial; safe only behind a trusted network/reverse proxy |

Current registry checks: verify_registry status OK for four entries; reconcile dry-run zero actions. Follow-up verification started local Qdrant and Neo4j successfully, loaded the local reranker, and passed 576 non-serving tests in `.venv_clean`. Historical environment failures and benchmark claims above remain historical and are not substituted with the current smoke results.

## H. Confirmed bugs and measurement defects

### H.1 Misleading reranker score contract (high evaluation/observability severity)

Location: retrieval/reranker/rerank_service.py:187-218; qdrant_retriever.py:268-275; frontend/components/chat/PipelineTrace.tsx:91-113.

Reproduce with lecture-wide retrieval and no global singleton: all vector scores are 1.0 and fallback rerank scores are 1.0. The field name implies cross-encoder output although no model ran. This does not necessarily break answer generation, but it invalidates score interpretation and can inflate downstream evaluation.

### H.2 Dependency metadata/runtime inconsistency (high reproducibility severity)

Full pytest stops at local/tests/test_d3_reranker.py with a transformers backend mapping error. pip check reports incompatible sentence-transformers/transformers/huggingface-hub/tokenizers. Duplicate Torch metadata causes pip and import versions to disagree. This blocks reliable reranker startup and full-suite verification.

### H.3 Historical report arithmetic/narrative errors (medium)

final_evaluation_report.md disagrees with raw report for pre Hit@1, pre Hit@3, and post Hit@1. The raw JSON and verification artifact support corrected values shown above.

### H.4 Citation metric name overstates guarantee (medium)

citation_completeness is an alias for provenance precision, not factual support. A 100% value must not be presented as citation faithfulness.

### H.5 RAGAS harness/API mismatch (medium)

The CLI requires source text and non-empty ground truth, but the current response schema only returns source metadata and the built-in test data has one empty-groundtruth sample. No valid RAGAS report exists.

### H.6 Public deployment exposure (high outside trusted localhost)

local Docker uses NEO4J_AUTH=none and exposes database ports. Production compose requires a password but publishes Neo4j/Qdrant/Ollama ports and shows no API authentication layer. This may be acceptable for localhost development, not an untrusted network.

## I. Unverified claims

- Current API process reranker state and model memory; the user’s process is not reachable from this shell.
- Fresh Qdrant/Neo4j retrieval, graph path evidence, and cross-lecture leakage.
- Whether ENABLE_AUTO_MODEL_RECOVERY=true succeeds with this exact broken dependency state; no model download was attempted.
- Full live execution of all six query classes.
- Current throughput, concurrency, memory, error rates.
- Human factual correctness of all historical answers.
- RAGAS faithfulness/relevancy/context precision.

## J. Prioritized correction plan

### P0 — correctness and observability

1. Expose reranker_active, model ID/version, device, and fallback reason in telemetry. Use a separate sampling_score and null/typed rerank score when inactive.
2. Reconcile the venv from one compatible lock set; eliminate duplicate Torch metadata; verify imports, pip check, and full pytest.
3. Make startup readiness/degraded mode explicit when the reranker is unavailable.
4. Return structured backend error status instead of embedding operational errors in answer text.

### P1 — trustworthy evaluation

1. Regenerate after environment/model repair with git commit, dependency versions, model hashes, provider/model, reranker state, package hashes, and raw ranked IDs/scores.
2. Generate narrative reports only from raw aggregates and add consistency checks.
3. Add a real dense-only artifact.
4. Add claim-level citation precision/recall, entailment, unsupported-claim rate, timestamp accuracy, answer completeness, and human factual review.
5. Expand to multi-lecture, independently-authored, unanswerable/adversarial, and route-balanced data.
6. Keep the circular graph test labeled as integration coverage; use held-out GraphRAG for quality claims.
7. Fix RAGAS with an internal context adapter/test-only endpoint, non-empty ground truth, and a multi-sample set.

### P2 — production/MLOps

1. Lock model artifacts and expose readiness for Qdrant, Neo4j, embedding, reranker, and LLM.
2. Add authentication, TLS/reverse proxy, network isolation, and non-public DB bindings; keep NEO4J_AUTH=none local only.
3. Produce real load reports with warmup, rate limits, p50/p95/p99, throughput, errors, and memory/CPU.
4. Add trace IDs and secret-free structured stage events.
5. Version datasets and package hashes with each run; block unproven dirty/mismatched runs.

## K. Final assessment

1. Actual RAG system: yes for the demonstrated online query; full live DB/reranker state was not reachable here.
2. Reranker: active and verified in the clean server after model installation; the existing port-8000 process must be restarted under `.venv_clean` to use it.
3. GraphRAG: traversal works, generalized quality is mixed (held-out structural recovery .52).
4. Answers: demonstrated answer is relevant; system-wide factual correctness is unproven.
5. Citations: metadata provenance is strong; claim-level faithfulness is unmeasured.
6. Retrieval metrics: historical raw aggregates are reproducible with corrected narrative values, but are single-lecture and run-specific.
7. Misleading perfect scores: live fallback 1.0 scores, citation completeness, chunk coverage, targeted routing, circular graph recovery, pairwise training MRR.
8. Before more training: stabilize dependencies and make reranker state/model provenance observable; then regenerate a valid baseline.
9. Technical interview: suitable as a qualified prototype demonstration, not as an unqualified production-quality claim.
10. Before production readiness: green reproducible environment, active model health, multi-lecture human-blind evaluation, claim-level faithfulness, independent GraphRAG, real load/error/memory results, and secured deployment.

## Commands and tests executed

- Redacted .env inspection and source tracing with rg, sed, numbered source views.
- Provider registry inspection reporting only provider/model/boolean key presence.
- scripts/verify_registry.py --json: four verified, zero problems, status OK.
- scripts/reconcile_registry.py dry-run: zero actions, zero destructive actions.
- Planner checks for factual, conceptual, summary, multi-hop, visual, and unanswerable examples.
- A clean `.venv_clean` was created from `local_requirements.txt`; `pip check` passed and imports resolved to torch 2.3.1+cpu, transformers 4.57.6, and sentence-transformers 3.3.1. The pre-existing `.venv` was not altered.
- In the clean environment, the non-serving suites (`agent/tests cloud/tests evaluation/tests retrieval/tests local/tests tests`) passed: 576 tests in 9.62 seconds. The full collected suite contains 624 tests; serving tests require the externally running services and were not claimed as passed.
- Qdrant and Neo4j Docker containers were started with the local compose file after explicit approval; escalated health checks returned `healthz check passed` and `neo4j_http_ok`. No collections or graph data were deleted or rebuilt.
- Direct clean-environment embedding plus Qdrant retrieval returned five lecture-filtered candidates with cosine scores 0.6449, 0.6156, 0.5939, 0.5892, and 0.5871, proving the stored vector data and clean embedding path are available.
- A temporary clean API on port 8001 was queried safely and then stopped. It returned `Insufficient evidence found in lecture` for an out-of-scope Global Data Plane question while the requested lecture package contains CS162 operating-systems material. This is correct grounding behavior.
- The already-running API on port 8000 returned a relevant-looking answer for the same out-of-scope wording, with `vector=0` and BM25-like fallback scores (6.667, 3.8271, 3.4693). This demonstrates a live configuration/runtime discrepancy that requires restarting the serving process under the clean environment before comparing quality.
- The configured generic `BAAI/bge-reranker-base` model is now present at `local_runtime/models/global_reranker/`, and the clean runtime loads it for benchmark runs.
- Direct inspection/recomputation of saved reports, question-level verification, routing, graph, and reproducibility artifacts.

### Remediation addendum (follow-up)

- Installed the configured generic `BAAI/bge-reranker-base` model at `local_runtime/models/global_reranker/` and enabled `ENABLE_AUTO_MODEL_RECOVERY=true` in the local `.env`.
- Restarted the API under `.venv_clean` and verified a live query reported `reranker_active=true`, `rerank_score_semantics=cross_encoder`, and changed cross-encoder ranking scores (for example 0.8221, 0.7951, 0.7377).
- Updated `retrieval/graph_retriever/neo4j_retriever.py` to read optional relationship metadata through `properties(rel)`, eliminating Neo4j unknown-property warnings for older graph imports while retaining confidence/provenance when present. A graph query returned route `graph_only` with 10 graph results and no property warnings.
- Added score-semantics fields to the API debug trace and frontend display so fallback and timeline scores cannot be mistaken for confidence. Frontend TypeScript validation passed.
- Added `.venv_clean/` to `.gitignore` to prevent committing the local verification environment.
- Regression verification after changes: 576 non-serving tests passed in 9.46 seconds; frontend `npx tsc --noEmit` passed.
- Remaining non-correctness warnings: numerous unregistered package directories require manual reconciliation, and the current Transformers stack emits a tokenizer regex warning during model load. Neither prevented the verified live query.
- Release-profile validation of `evaluation/datasets/cs162_lecture1_qa_50.json` correctly fails: 50 samples from one lecture, one route, only two hard questions, and sparse definition/summary categories. This benchmark must not be labeled release-grade.
- A one-request load-test smoke attempt exceeded a 20-second timeout and produced no report; no throughput claim is made. The full 100/500/1000 concurrency test was intentionally not launched because it would generate substantial live-provider traffic.
- Added a Groq-grounded evaluation draft at `evaluation/datasets/multilecture_groq_eval_draft.json`: 160 records across three registered lectures, with vector, graph, visual, and five unanswerable examples. The draft includes source chunk provenance and explicit `review_status` values; it is not human-reviewed and intentionally fails release validation until reviewed.
- Groq-backed provisional evaluation ran successfully on 12 samples (12/12 completed) using the live Qdrant, Neo4j, embedding, reranker, and configured provider. The report is under `evaluation/reports/provisional_groq_20261010/`; it is diagnostic only. It measured mean Hit@5 0.9167, MRR@5 0.6486, answer F1 0.2898, citation provenance precision 1.0 (structural), and mean total latency 13.01 seconds. These numbers must not be presented as final quality claims because the labels are not human-reviewed and the slice is small.
- The 1/2/4-concurrency smoke load report now exists at `evaluation/reports/load_test_report.csv`: concurrency 1 and 2 completed with zero errors, while concurrency 4 had 2/4 errors and mean latency 35.45 seconds. This is a real stability signal, not a production benchmark; provider rate limiting and shared model/LLM contention need investigation before claiming concurrency support.

### 45-question three-lecture smoke evaluation (corrected)

- The fixed smoke dataset is `evaluation/datasets/multilecture_smoke_45.json`: 45 records, exactly 15 per registered lecture, with factual, conceptual, visual, graph, definition, and two unanswerable cases. It is explicitly `provisional_not_human_reviewed`; it is a fast integration smoke test, not a release benchmark.
- The first full run completed 45/45 samples successfully using the live Qdrant, Neo4j, CPU embedding model, local cross-encoder, and configured Groq backend. Raw results are preserved in `evaluation/reports/smoke_groq_20261010/`.
- During review, a metric defect was confirmed: the retrieval and citation functions treated an empty expected-evidence set as a perfect score. The benchmark runner now excludes unanswerable cases from evidence-based retrieval/reranker/citation denominators and reports refusal diagnostics separately. The corrected report is `evaluation/reports/smoke_groq_20261010_fixed/`.
- Corrected aggregate values (answerable denominator `n=43` where shown): Hit@5 **0.8605 (43)**, Recall@5 **0.8488 (43)**, Precision@5 **0.1721 (43)**, MRR@5 **0.7333 (43)**, unbounded MRR **0.7438 (43)**, nDCG@5 **0.7595 (43)**, reranker ranking quality **0.7644 (43)**, citation coverage **0.7674 (43)**, answer F1 **0.2667 (45)**, keyword recall **0.7540 (45)**, lexical Jaccard **0.1935 (45)**, citation provenance precision **1.0000 (45; structural only)**, routing accuracy **0.8889 (45)**, visual routing accuracy **0.9111 (45)**, and mean total latency **17.32 seconds (45)**.
- The two unanswerable samples were correctly refused according to the marker-based smoke diagnostic (`refusal_correct=1.0, n=2`), but this is not a human factual judgment. Their retrieval/citation metrics are intentionally `n/a`, not zero or one.
- Graph questions are the weak slice: Hit@5 **0.3333**, Recall@5 **0.1667**, MRR@5 **0.1667**, and citation coverage **0.0** over three samples. This is evidence that graph retrieval/evidence mapping needs work; it is not hidden by the unanswerable correction.
- Answer quality remains materially below retrieval quality (F1 **0.2667** versus Recall@5 **0.8488**), so high retrieval numbers must not be presented as answer correctness. The low F1 is plausible for automatically generated reference answers and needs human-reviewed references before model comparisons.
- A duplicate full Groq rerun was stopped after the provider returned HTTP 429/token-per-day throttling. The corrected report is therefore a deterministic recalculation of the completed raw run, not a second generation run; no new response data were substituted.

Only this Markdown report was created. No application code, stored data, model, database, environment variable, or secret was modified.

