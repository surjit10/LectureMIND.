# Minimal LectureMind MLOps workflow

LectureMind uses a deliberately small, explainable MLOps workflow:

1. Validate the benchmark structure before running it.
2. Run the real retrieval/reranking/generation workflow when explicitly requested.
3. Save JSON/CSV/Markdown results with a run manifest, Git state, dependency versions,
   dataset SHA-256, and lecture-package hashes.
4. Render a read-only HTML dashboard from those artifacts.

Safe validation and dashboard generation (does not call an LLM):

```bash
./.venv_clean/bin/python scripts/run_mlops_smoke.py
```

Run the live smoke benchmark only when provider quota and local Qdrant/Neo4j are
available:

```bash
./.venv_clean/bin/python scripts/run_mlops_smoke.py --run
```

This is an MLOps foundation, not a complete production platform. It intentionally
does not claim automated model promotion, drift detection, alerting, or rollback.
