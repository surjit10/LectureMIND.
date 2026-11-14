"""Run the real retrieval + rerank + context pipeline (no LLM) for one code tree.

Usage: python scripts/compare_retrieval_run.py <code_tree> <dataset.json> <out.jsonl>

<code_tree> is the checkout whose code is imported (e.g. a git worktree of an
older commit, or this repo). Models and data always come from this repo.

Example (old vs current code):
    git worktree add --detach /tmp/old_tree <commit> && cp .env /tmp/old_tree/
    python scripts/compare_retrieval_run.py /tmp/old_tree evaluation/datasets/multilecture_smoke_45.json old.jsonl
    python scripts/compare_retrieval_run.py . evaluation/datasets/multilecture_smoke_45.json new.jsonl
    python scripts/compare_retrieval_analyze.py old.jsonl new.jsonl comparison.json

Mirrors agent/langgraph/workflow.py steps 1-3 exactly (planner -> graph
retrieval for graph routes -> vector retrieval for every route -> reranker
node incl. BM25/RRF + cross-encoder + ContextBuilder), against the live local
Qdrant/Neo4j. Answer generation (step 4) is skipped, so no provider quota.
"""
import json
import os
import sys
import time
from pathlib import Path

TREE = Path(sys.argv[1]).resolve()
DATASET = Path(sys.argv[2]).resolve()
OUT = Path(sys.argv[3]).resolve()
MAIN_REPO = Path(__file__).resolve().parents[1]
RERANKER_DIR = MAIN_REPO / "local_runtime/models/global_reranker"

os.chdir(TREE)
sys.path.insert(0, str(TREE))

import logging
logging.disable(logging.WARNING)

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from agent.dspy.planner import QueryPlanner
from agent.langgraph.nodes.graph_retriever import graph_retriever_node
from agent.langgraph.nodes.reranker import reranker_node
from agent.langgraph.nodes.vector_retriever import vector_retriever_node
from local.loaders.reranker_loader import _load_cross_encoder
from local.services.reranker_service import RerankerService
from schemas.enums import RetrievalRoute
import retrieval.reranker.rerank_service as rs

qdrant = QdrantClient(url="http://localhost:6333")
driver = GraphDatabase.driver("bolt://localhost:7687")
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
service = RerankerService(_load_cross_encoder(RERANKER_DIR, quantize=False))
rs._GLOBAL_RERANKER_SERVICE = service
planner = QueryPlanner()

# Record the exact candidate pool rerank() scores (after any BM25/RRF fusion).
_orig_extract = rs._extract_passages
_captured = {}


def _spy(graph_results, vector_results):
    out = _orig_extract(graph_results, vector_results)
    _captured["pool"] = [(r.get("payload", r).get("chunk_id") or r.get("chunk_id", "")) for r in out]
    return out


rs._extract_passages = _spy


def cid(r):
    p = r.get("payload", r)
    return p.get("chunk_id") or r.get("chunk_id", "")


def run_one(query, lecture_id):
    _captured.clear()
    plan = planner.plan_full(query)
    state = {"query": query, "lecture_id": lecture_id, "query_plan": plan,
             "graph_results": [], "vector_results": []}
    t0 = time.perf_counter()
    if plan.retrieval_route in (RetrievalRoute.graph_only, RetrievalRoute.graph_and_vector):
        state.update(graph_retriever_node(state, driver=driver))
    state.update(vector_retriever_node(state, qdrant_client=qdrant, embedding_model=embedder))
    t1 = time.perf_counter()
    state.update(reranker_node(state, reranker_service=service))
    t2 = time.perf_counter()
    reranked = state["reranked_results"]
    ctx = state["final_context"]
    return {
        "route": plan.retrieval_route.name,
        "intent": plan.intent,
        "is_lecture_wide": plan.is_lecture_wide,
        "graph_rows": len(state["graph_results"]),
        "graph_lines": sum(1 for line in ctx.splitlines() if line.startswith("[Graph]")),
        "pool_ids": _captured.get("pool", []),
        "reranked_ids": [cid(r) for r in reranked],
        "context_ids": [cid(r) for r in reranked if r.get("in_context")],
        "context_timestamps": sorted(float(r.get("payload", r).get("timestamp", 0) or 0)
                                     for r in reranked if r.get("in_context")),
        "context_chars": len(ctx),
        "retrieval_s": t1 - t0,
        "rerank_s": t2 - t1,
    }


samples = json.loads(DATASET.read_text())
lectures = sorted({s["lecture_id"] for s in samples})
SUMMARY_PROMPTS = ["Summarize the lecture", "What are the key takeaways?",
                   "Give me an overview of today's lecture"]

# Warm-up per lecture (BM25 index build, model warm caches) — not recorded.
for lid in lectures:
    run_one("warm up query", lid)

with OUT.open("w") as fh:
    for i, s in enumerate(samples):
        rec = run_one(s["query"], s["lecture_id"])
        rec.update({"kind": "benchmark", "idx": i, "lecture_id": s["lecture_id"], "query": s["query"],
                    "question_type": s.get("question_type"), "expected_route": s.get("expected_route"),
                    "answerable": s.get("answerable", True), "expected_chunk_ids": s.get("expected_chunk_ids", [])})
        fh.write(json.dumps(rec) + "\n")
    for lid in lectures:
        for q in SUMMARY_PROMPTS:
            rec = run_one(q, lid)
            rec.update({"kind": "summary", "lecture_id": lid, "query": q})
            fh.write(json.dumps(rec) + "\n")
print(f"wrote {OUT}")
driver.close()
