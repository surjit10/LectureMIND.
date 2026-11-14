"""Component ablation + context-budget experiment (no LLM calls).

Usage:
    python scripts/run_ablation.py [--dataset evaluation/datasets/multilecture_smoke_45.json]
                                   [--out-dir evaluation/reports/ablation_<date>]

For every question, retrieval runs once (planner -> graph retrieval for graph
routes -> dense retrieval), then each configuration rebuilds the reranked list
and the LLM context with the production rerank()/ContextBuilder code:

  Ablation (production budgets)
    dense_only            dense order, no BM25, no cross-encoder, no graph
    dense_bm25            + BM25 via RRF (no cross-encoder, no graph)
    dense_bm25_rerank     + cross-encoder (no graph)
    full_pipeline         + graph context (= production behaviour)

  Context-budget variants of the full pipeline
    budget<B>_graph<S>    point-query budget B chars, graph share S of budget

Cross-encoder scores are cached per (query, passage), so the reranker runs once
per question. Requires live Qdrant + Neo4j, bge-large-en-v1.5 and the reranker
in local_runtime/models/global_reranker.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import logging
logging.disable(logging.WARNING)

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from agent.planner.planner import QueryPlanner
from agent.pipeline.nodes.graph_retriever import graph_retriever_node
from agent.pipeline.nodes.vector_retriever import vector_retriever_node
from evaluation.metrics import retrieval_metrics as rm
from local.loaders.reranker_loader import _load_cross_encoder
from local.services.reranker_service import RerankerService
from schemas.enums import RetrievalRoute
import retrieval.reranker.rerank_service as rs


class CachedService:
    """Cross-encoder with a per-(query, passage) score cache."""

    def __init__(self, service):
        self._service, self._cache = service, {}

    def score_pairs(self, query, passages):
        todo = [p for p in passages if (query, p) not in self._cache]
        if todo:
            for p, s in self._service.score_pairs(query, todo):
                self._cache[(query, p)] = s
        return sorted(((p, self._cache[(query, p)]) for p in passages), key=lambda x: x[1], reverse=True)


class OrderService:
    """'No reranker': keeps the incoming candidate order (dense or RRF)."""

    def score_pairs(self, query, passages):
        return [(p, 1.0 - i * 1e-4) for i, p in enumerate(passages)]


def cid(r):
    p = r.get("payload", r)
    return p.get("chunk_id") or r.get("chunk_id", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="evaluation/datasets/multilecture_smoke_45.json")
    ap.add_argument("--out-dir", default="evaluation/reports/ablation_latest")
    args = ap.parse_args()
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    qdrant = QdrantClient(url="http://localhost:6333")
    driver = GraphDatabase.driver("bolt://localhost:7687")
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    ce = CachedService(RerankerService(_load_cross_encoder(REPO / "local_runtime/models/global_reranker", quantize=False)))
    order = OrderService()
    rs._GLOBAL_RERANKER_SERVICE = None  # use the service passed per call
    planner = QueryPlanner()

    configs = {
        "dense_only": dict(hybrid=False, service=order, graph=False, budget=None, share=None),
        "dense_bm25": dict(hybrid=True, service=order, graph=False, budget=None, share=None),
        "dense_bm25_rerank": dict(hybrid=True, service=ce, graph=False, budget=None, share=None),
        "full_pipeline": dict(hybrid=True, service=ce, graph=True, budget=None, share=None),
    }
    for budget in (4000, 6000):
        for share in (0.5, 0.25):
            configs[f"budget{budget}_graph{share}"] = dict(hybrid=True, service=ce, graph=True,
                                                           budget=budget, share=share)

    samples = json.loads((REPO / args.dataset).read_text())
    rows = []
    default_share, default_cap = rs.GRAPH_CONTEXT_SHARE, rs.MAX_GRAPH_CONTEXT_CHARS
    for i, s in enumerate(samples):
        plan = planner.plan_full(s["query"])
        state = {"query": s["query"], "lecture_id": s["lecture_id"], "query_plan": plan,
                 "graph_results": [], "vector_results": []}
        if plan.retrieval_route in (RetrievalRoute.graph_only, RetrievalRoute.graph_and_vector):
            state.update(graph_retriever_node(state, driver=driver))
        state.update(vector_retriever_node(state, qdrant_client=qdrant, embedding_model=embedder))
        row = {"idx": i, "query": s["query"], "lecture_id": s["lecture_id"],
               "question_type": s.get("question_type"), "answerable": s.get("answerable", True),
               "expected_chunk_ids": s.get("expected_chunk_ids", []), "route": plan.retrieval_route.name,
               "configs": {}}
        for name, c in configs.items():
            if c["share"] is not None:
                rs.GRAPH_CONTEXT_SHARE = c["share"]
                rs.MAX_GRAPH_CONTEXT_CHARS = int(c["budget"] * c["share"])
            else:
                rs.GRAPH_CONTEXT_SHARE, rs.MAX_GRAPH_CONTEXT_CHARS = default_share, default_cap
            reranked, ctx = rs.rerank(
                s["query"], state["graph_results"] if c["graph"] else [], state["vector_results"],
                reranker_service=c["service"], is_lecture_wide=plan.is_lecture_wide,
                need_visual=plan.need_visual,
                char_budget=c["budget"] if c["budget"] is not None else plan.context_budget,
                lecture_id=s["lecture_id"], enable_hybrid=c["hybrid"],
            )
            row["configs"][name] = {
                "ranked_ids": [cid(r) for r in reranked],
                "context_ids": [cid(r) for r in reranked if r.get("in_context")],
                "context_chars": len(ctx),
                "graph_lines": sum(1 for line in ctx.splitlines() if line.startswith("[Graph]")),
            }
        rs.GRAPH_CONTEXT_SHARE, rs.MAX_GRAPH_CONTEXT_CHARS = default_share, default_cap
        rows.append(row)
        print(f"{i + 1}/{len(samples)}", flush=True)

    (out_dir / "per_question.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ans = [r for r in rows if r["answerable"]]
    summary = {}
    for name in configs:
        def m(f):
            return st.mean(f(r["configs"][name], r["expected_chunk_ids"]) for r in ans)
        graph_rows = [r for r in rows if r["route"] != "vector_only"]
        summary[name] = {
            "hit_at_1": m(lambda c, g: rm.calculate_hit_at_k(g, c["ranked_ids"], 1)),
            "hit_at_5": m(lambda c, g: rm.calculate_hit_at_k(g, c["ranked_ids"], 5)),
            "mrr_at_5": m(lambda c, g: rm.calculate_mrr(g, c["ranked_ids"], k=5)),
            "context_recall": m(lambda c, g: rm.calculate_context_recall(g, c["context_ids"])),
            "context_recall_graph_routed": (st.mean(rm.calculate_context_recall(r["expected_chunk_ids"], r["configs"][name]["context_ids"])
                                                    for r in graph_rows if r["answerable"]) if graph_rows else None),
            "mean_context_chars": st.mean(r["configs"][name]["context_chars"] for r in rows),
            "mean_graph_lines_graph_routed": st.mean(r["configs"][name]["graph_lines"] for r in graph_rows) if graph_rows else 0.0,
        }
    meta = {"dataset": args.dataset, "n_questions": len(rows), "n_answerable": len(ans),
            "n_graph_routed": sum(r["route"] != "vector_only" for r in rows),
            "n_graph_routed_answerable": sum(r["route"] != "vector_only" and r["answerable"] for r in rows)}
    (out_dir / "ablation.json").write_text(json.dumps({"meta": meta, "configs": summary}, indent=2))
    print(json.dumps(meta))
    print(f"| config | Hit@1 | Hit@5 | MRR@5 | context recall | ctx recall (graph-routed) | mean ctx chars | graph lines |")
    for name, v in summary.items():
        print(f"| {name} | {v['hit_at_1']:.3f} | {v['hit_at_5']:.3f} | {v['mrr_at_5']:.3f} | {v['context_recall']:.3f} | "
              f"{v['context_recall_graph_routed']:.3f} | {v['mean_context_chars']:.0f} | {v['mean_graph_lines_graph_routed']:.1f} |")
    driver.close()


if __name__ == "__main__":
    main()
