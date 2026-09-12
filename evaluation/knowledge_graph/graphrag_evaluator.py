# evaluation/knowledge_graph/graphrag_evaluator.py
# Fair, reproducible downstream GraphRAG evaluation.
#
# Compares:
#   1. Vector / Lexical RAG (BM25 term-frequency saturation over chunks)
#   2. Graph-Only RAG (Entity recognition + 1-hop and 2-hop Knowledge Graph traversal)
#   3. Hybrid RAG (Reciprocal Rank Fusion of Vector + Graph candidates)
#
# Evaluated across the exact same benchmark queries and expected gold chunks.
# Computes:
#   - Hit@K (K=1, 3, 5)
#   - Recall@K (K=1, 3, 5)
#   - Precision@K (K=1, 3, 5)
#   - MRR (Mean Reciprocal Rank)

import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# BM25 Constants
K1 = 1.5
B = 0.75
RRF_K = 60


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 2]


class SimpleBM25:
    """Self-contained Okapi BM25 index over lecture chunks."""

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.doc_tokens = []
        self.doc_lens = []
        self.df = Counter()
        self.N = len(chunks)

        for c in chunks:
            full_text = (c.get("text") or c.get("transcript") or "") + " " + (c.get("ocr_text") or "")
            toks = _tokenize(full_text)
            self.doc_tokens.append(toks)
            self.doc_lens.append(len(toks))
            for term in set(toks):
                self.df[term] += 1

        self.avg_dl = sum(self.doc_lens) / self.N if self.N > 0 else 1.0

    def query(self, query_text: str) -> List[Tuple[str, float]]:
        q_toks = _tokenize(query_text)
        scores = [0.0] * self.N

        for term in q_toks:
            if term not in self.df:
                continue
            df_val = self.df[term]
            idf = math.log((self.N - df_val + 0.5) / (df_val + 0.5) + 1.0)

            for i, doc_toks in enumerate(self.doc_tokens):
                tf = doc_toks.count(term)
                if tf == 0:
                    continue
                num = tf * (K1 + 1)
                denom = tf + K1 * (1 - B + B * (self.doc_lens[i] / self.avg_dl))
                scores[i] += idf * (num / denom)

        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked


class SimpleGraphRetriever:
    """Self-contained KG retriever using entity recognition & multi-hop traversal."""

    def __init__(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
    ):
        self.entity_map = {e["entity_id"]: e["name"] for e in entities}
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        
        # Build entity -> chunks map
        self.entity_chunks: Dict[str, Set[str]] = {e["entity_id"]: set() for e in entities}
        for e in entities:
            eid = e["entity_id"]
            ename_low = e["name"].lower()
            for c in chunks:
                txt = ((c.get("text") or "") + " " + (c.get("ocr_text") or "")).lower()
                if ename_low in txt:
                    self.entity_chunks[eid].add(c["chunk_id"])

        # Build graph adjacency
        self.adj: Dict[str, List[Tuple[str, str]]] = {}  # eid -> [(neighbor_eid, rel_type), ...]
        for r in relations:
            src = r.get("source_entity_id") or r.get("source_id")
            tgt = r.get("target_entity_id") or r.get("target_id")
            rel = r.get("relation") or r.get("relation_type", "")
            if src and tgt:
                self.adj.setdefault(src, []).append((tgt, rel))
                self.adj.setdefault(tgt, []).append((src, rel))

    def query(self, query_text: str) -> List[Tuple[str, float]]:
        q_low = query_text.lower()
        matched_eids = []

        # 1. Match entities mentioned in query
        for eid, name in self.entity_map.items():
            if name.lower() in q_low and len(name) > 3:
                matched_eids.append(eid)

        # 2. Score chunks based on 0-hop (direct), 1-hop, and 2-hop graph distance
        chunk_scores = {cid: 0.0 for cid in self.chunk_ids}

        for eid in matched_eids:
            # 0-hop: chunks directly containing the entity
            for cid in self.entity_chunks.get(eid, set()):
                chunk_scores[cid] += 3.0

            # 1-hop: neighbors in the knowledge graph
            for neighbor_eid, _ in self.adj.get(eid, []):
                for cid in self.entity_chunks.get(neighbor_eid, set()):
                    chunk_scores[cid] += 1.5

                # 2-hop: neighbors of neighbors
                for n2_eid, _ in self.adj.get(neighbor_eid, []):
                    if n2_eid != eid:
                        for cid in self.entity_chunks.get(n2_eid, set()):
                            chunk_scores[cid] += 0.5

        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
) -> List[Tuple[str, float]]:
    """Combine ranked candidate lists using Reciprocal Rank Fusion (RRF)."""
    scores: Dict[str, float] = {}
    for rlist in ranked_lists:
        for rank, (cid, _) in enumerate(rlist, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


@dataclass
class RetrievalMetrics:
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0


@dataclass
class GraphRAGEvaluationResult:
    total_queries: int
    bm25_metrics: Dict[str, float]
    graph_rag_metrics: Dict[str, float]
    hybrid_rag_metrics: Dict[str, float]
    category_breakdown: Dict[str, Any]
    vector_rag_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Maintain vector_rag_metrics alias for backward compatibility
        if not self.vector_rag_metrics and self.bm25_metrics:
            self.vector_rag_metrics = self.bm25_metrics
        elif not self.bm25_metrics and self.vector_rag_metrics:
            self.bm25_metrics = self.vector_rag_metrics


def _compute_metrics(
    query_results: List[List[str]],
    expected_chunks_list: List[List[str]],
) -> Dict[str, float]:
    hits = {1: 0, 3: 0, 5: 0}
    recalls = {1: 0.0, 3: 0.0, 5: 0.0}
    precisions = {1: 0.0, 3: 0.0, 5: 0.0}
    total_rr = 0.0
    N = len(query_results)

    for retrieved, gold in zip(query_results, expected_chunks_list):
        gold_set = set(gold)
        if not gold_set:
            continue

        # MRR
        rr = 0.0
        for rank, cid in enumerate(retrieved, start=1):
            if cid in gold_set:
                rr = 1.0 / rank
                break
        total_rr += rr

        # Top K
        for k in [1, 3, 5]:
            top_k = retrieved[:k]
            matched = sum(1 for cid in top_k if cid in gold_set)
            if matched > 0:
                hits[k] += 1
            recalls[k] += (matched / len(gold_set))
            precisions[k] += (matched / k)

    return {
        "Hit@1": round(hits[1] / N, 4) if N > 0 else 0.0,
        "Hit@3": round(hits[3] / N, 4) if N > 0 else 0.0,
        "Hit@5": round(hits[5] / N, 4) if N > 0 else 0.0,
        "Recall@1": round(recalls[1] / N, 4) if N > 0 else 0.0,
        "Recall@3": round(recalls[3] / N, 4) if N > 0 else 0.0,
        "Recall@5": round(recalls[5] / N, 4) if N > 0 else 0.0,
        "Precision@1": round(precisions[1] / N, 4) if N > 0 else 0.0,
        "Precision@3": round(precisions[3] / N, 4) if N > 0 else 0.0,
        "Precision@5": round(precisions[5] / N, 4) if N > 0 else 0.0,
        "MRR": round(total_rr / N, 4) if N > 0 else 0.0,
    }


def evaluate_graphrag(
    chunks: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    benchmark_dataset: List[Dict[str, Any]],
) -> GraphRAGEvaluationResult:
    """
    Run the fair downstream comparison between Vector, Graph, and Hybrid RAG.
    """
    vector_retriever = SimpleBM25(chunks)
    graph_retriever = SimpleGraphRetriever(entities, relations, chunks)

    vector_runs: List[List[str]] = []
    graph_runs: List[List[str]] = []
    hybrid_runs: List[List[str]] = []
    gold_chunks_list: List[List[str]] = []
    categories: Dict[str, List[int]] = {}

    for idx, sample in enumerate(benchmark_dataset):
        q = sample["query"]
        gold = sample.get("expected_chunk_ids", [])
        gold_chunks_list.append(gold)
        
        q_type = sample.get("question_type", "general")
        categories.setdefault(q_type, []).append(idx)

        # 1. Vector RAG
        v_ranked = vector_retriever.query(q)
        vector_runs.append([cid for cid, _ in v_ranked])

        # 2. Graph RAG
        g_ranked = graph_retriever.query(q)
        graph_runs.append([cid for cid, _ in g_ranked])

        # 3. Hybrid RAG (RRF)
        h_ranked = _reciprocal_rank_fusion([v_ranked, g_ranked])
        hybrid_runs.append([cid for cid, _ in h_ranked])

    # Overall Metrics
    v_metrics = _compute_metrics(vector_runs, gold_chunks_list)
    g_metrics = _compute_metrics(graph_runs, gold_chunks_list)
    h_metrics = _compute_metrics(hybrid_runs, gold_chunks_list)

    # Category-specific Breakdown
    category_summary: Dict[str, Any] = {}
    for cat, idxs in categories.items():
        sub_v = [vector_runs[i] for i in idxs]
        sub_g = [graph_runs[i] for i in idxs]
        sub_h = [hybrid_runs[i] for i in idxs]
        sub_gold = [gold_chunks_list[i] for i in idxs]
        category_summary[cat] = {
            "query_count": len(idxs),
            "bm25_mrr": _compute_metrics(sub_v, sub_gold)["MRR"],
            "vector_mrr": _compute_metrics(sub_v, sub_gold)["MRR"],  # compatibility alias
            "graph_mrr": _compute_metrics(sub_g, sub_gold)["MRR"],
            "hybrid_mrr": _compute_metrics(sub_h, sub_gold)["MRR"],
            "hybrid_recall@5": _compute_metrics(sub_h, sub_gold)["Recall@5"],
        }

    return GraphRAGEvaluationResult(
        total_queries=len(benchmark_dataset),
        bm25_metrics=v_metrics,
        vector_rag_metrics=v_metrics,
        graph_rag_metrics=g_metrics,
        hybrid_rag_metrics=h_metrics,
        category_breakdown=category_summary,
    )
