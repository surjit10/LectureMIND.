# evaluation/run_heldout_graph_evaluation.py
"""
Authentic Held-Out GraphRAG and Multi-Hop Traversal Evaluation Runner.

Evaluates LectureMIND on an independently created, held-out CS162 graph benchmark:
  1. Routing / Planner accuracy
  2. Structural Graph Traversal (1-hop, 2-hop, 3-hop path recovery in live Neo4j)
  3. Downstream Chunk Retrieval (BM25, Graph, Hybrid RRF Hit@K & Recall)
  4. Negative Question Resistance (false positive path detection, refusal verification)
  5. Final Answer Generation Correctness (Token F1, Keyword Recall, Refusal Accuracy)

Produces:
  - evaluation/outputs/graph_heldout_evaluation.json
  - evaluation/outputs/graph_heldout_evaluation.md
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.dspy.planner import QueryPlanner
from agent.langgraph.nodes.answer_generator import _generate_answer
from evaluation.knowledge_graph.graphrag_evaluator import (
    SimpleBM25,
    SimpleGraphRetriever,
    _compute_metrics,
    _reciprocal_rank_fusion,
)
from evaluation.metrics.answer_metrics import (
    calculate_answer_f1,
    calculate_answer_similarity,
    calculate_keyword_recall,
)
from local.llm.provider_registry import get_provider_registry
from retrieval.graph_retriever.neo4j_retriever import retrieve_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graph_heldout_eval")


def run_evaluation(
    benchmark_path: str = "evaluation/datasets/cs162_lecture1_graph_qa_heldout.json",
    packages_dir: str = "data/packages/lecture_092f861b",
    lecture_id: str = "lecture_092f861b",
) -> Dict[str, Any]:
    bench_file = Path(benchmark_path)
    pkg_path = Path(packages_dir)

    with open(bench_file, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    with open(pkg_path / "entities.json", "r", encoding="utf-8") as f:
        entities = json.load(f)
    with open(pkg_path / "relations.json", "r", encoding="utf-8") as f:
        relations = json.load(f)
    with open(pkg_path / "multimodal_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    chunk_map = {c["chunk_id"]: c for c in chunks}

    # Initialize components
    planner = QueryPlanner()
    bm25 = SimpleBM25(chunks)
    graph_sim = SimpleGraphRetriever(entities, relations, chunks)
    llm_backend = get_provider_registry().get_active_backend()

    # Track metrics by category
    hop_stats = {
        1: {"total": 0, "path_hits": 0, "hit5_hits": 0, "ans_correct": 0, "f1_scores": []},
        2: {"total": 0, "path_hits": 0, "hit5_hits": 0, "ans_correct": 0, "f1_scores": []},
        3: {"total": 0, "path_hits": 0, "hit5_hits": 0, "ans_correct": 0, "f1_scores": []},
    }
    neg_stats = {
        "total": 0,
        "false_positive_paths": 0,
        "correct_refusals": 0,
        "hallucinated_answers": 0,
    }

    bm25_runs = []
    graph_runs = []
    hybrid_runs = []
    gold_chunk_runs = []

    eval_items = []

    logger.info("Starting held-out GraphRAG evaluation on %d questions...", len(benchmark))

    for item in benchmark:
        qid = item["question_id"]
        q = item["question"]
        hop = item["hop_count"]
        ans = item["answerable"]
        qtype = item["question_type"]
        gold_ans = item.get("gold_answer", "")
        gold_ents = item.get("gold_entities", [])
        gold_rels = item.get("gold_relations", [])
        gold_cids = item.get("gold_source_chunk_ids", [])

        # 1. Routing / Planning
        plan = planner.plan_full(q)
        route_str = plan.retrieval_route.value if hasattr(plan.retrieval_route, "value") else str(plan.retrieval_route)

        # 2. Structural Graph Traversal (Live Neo4j)
        try:
            neo4j_edges = retrieve_graph(q, lecture_id=lecture_id, max_hops=3)
        except Exception as exc:
            logger.warning("[%s] retrieve_graph failed: %s", qid, exc)
            neo4j_edges = []

        path_recovered = False
        false_positive_path = False
        recovered_rels_count = 0

        if ans:
            hop_stats[hop]["total"] += 1
            # Check edge-by-edge match for gold relations
            for s, rel, d in gold_rels:
                s_low, rel_low, d_low = s.lower(), rel.lower(), d.lower()
                found = False
                for r in neo4j_edges:
                    s_name = (r.get("start_name") or "").lower()
                    d_name = (r.get("related_name") or "").lower()
                    r_types = [t.lower() for t in (r.get("rel_types") or [])]
                    if (s_name == s_low and d_name == d_low) or (s_name == d_low and d_name == s_low):
                        if rel_low in r_types or any(rel_low in t for t in r_types):
                            found = True
                            break
                if found:
                    recovered_rels_count += 1
            if len(gold_rels) > 0 and recovered_rels_count == len(gold_rels):
                path_recovered = True
                hop_stats[hop]["path_hits"] += 1
        else:
            neg_stats["total"] += 1
            # For negative question: does the graph return a false connecting edge between negative query subjects?
            ents_low = [e.lower() for e in gold_ents]
            for r in neo4j_edges:
                s_name = (r.get("start_name") or "").lower()
                d_name = (r.get("related_name") or "").lower()
                if len(ents_low) >= 2 and s_name in ents_low and d_name in ents_low and s_name != d_name:
                    false_positive_path = True
                    break
            if false_positive_path:
                neg_stats["false_positive_paths"] += 1

        # 3. Downstream Chunk Retrieval
        v_res = [cid for cid, _ in bm25.query(q)]
        g_res = [cid for cid, _ in graph_sim.query(q)]
        h_res = [cid for cid, _ in _reciprocal_rank_fusion([bm25.query(q), graph_sim.query(q)])]

        hit1 = 0
        hit3 = 0
        hit5 = 0
        if ans:
            bm25_runs.append(v_res)
            graph_runs.append(g_res)
            hybrid_runs.append(h_res)
            gold_chunk_runs.append(gold_cids)

            hit1 = 1 if any(cid in gold_cids for cid in h_res[:1]) else 0
            hit3 = 1 if any(cid in gold_cids for cid in h_res[:3]) else 0
            hit5 = 1 if any(cid in gold_cids for cid in h_res[:5]) else 0
            if hit5:
                hop_stats[hop]["hit5_hits"] += 1

        # 4. Context Assembly & Answer Generation
        # Construct grounded context from top retrieved chunks and graph edges
        context_snippets = []
        for cid in h_res[:3]:
            c = chunk_map.get(cid)
            if c:
                text = (c.get("transcript") or c.get("ocr_text") or "").strip()
                if text:
                    context_snippets.append(f"[Chunk {cid}] {text}")

        # Add graph evidence
        if neo4j_edges:
            top_edges = neo4j_edges[:5]
            edge_strs = [
                f"{e.get('start_name')} -[{'|'.join(e.get('rel_types') or [])}]-> {e.get('related_name')}"
                for e in top_edges
            ]
            context_snippets.append("Graph Relationships:\n" + "\n".join(edge_strs))

        final_context = "\n\n".join(context_snippets)

        # Check if we have a checkpointed answer for this qid
        cached_item = None
        checkpoint_path = Path("evaluation/outputs/graph_heldout_checkpoint.json")
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as ckf:
                    cache_data = json.load(ckf)
                    cached_item = cache_data.get(qid)
            except Exception:
                pass

        if cached_item and cached_item.get("generated_answer"):
            gen_answer = cached_item["generated_answer"]
            meta = {}
            logger.info("[%s] Reusing checkpointed answer.", qid)
        else:
            gen_answer, meta = _generate_answer(
                query=q,
                context=final_context,
                backend=llm_backend,
                answer_style="concise",
            )
            # Save to checkpoint
            try:
                cache_data = {}
                if checkpoint_path.exists():
                    with open(checkpoint_path, "r", encoding="utf-8") as ckf:
                        cache_data = json.load(ckf)
                cache_data[qid] = {"generated_answer": gen_answer}
                with open(checkpoint_path, "w", encoding="utf-8") as ckf:
                    json.dump(cache_data, ckf, indent=2)
            except Exception:
                pass

        # 5. Answer Evaluation
        refusal_keywords = [
            "insufficient evidence", "not mentioned", "not supported",
            "no evidence", "not discussed", "does not state",
            "not directly related", "not a prerequisite", "no direct"
        ]
        is_refusal = any(kw in gen_answer.lower() for kw in refusal_keywords)

        ans_f1 = 0.0
        kw_recall = 0.0
        similarity = 0.0
        ans_correct = False

        if ans:
            ans_f1 = calculate_answer_f1(gold_ans, gen_answer)
            kw_recall = calculate_keyword_recall(gold_ents, gen_answer)
            similarity = calculate_answer_similarity(gold_ans, gen_answer)
            hop_stats[hop]["f1_scores"].append(ans_f1)
            # Answer is deemed correct if semantic similarity/keyword recall shows factual alignment
            # and is not an ungrounded refusal when evidence was present
            if (kw_recall >= 0.5 or ans_f1 >= 0.25) and not is_refusal:
                ans_correct = True
                hop_stats[hop]["ans_correct"] += 1
        else:
            # Negative question: correct if LLM explicitly refused or noted lack of evidence
            if is_refusal or "insufficient" in gen_answer.lower():
                ans_correct = True
                neg_stats["correct_refusals"] += 1
            else:
                neg_stats["hallucinated_answers"] += 1

        eval_items.append({
            "question_id": qid,
            "question": q,
            "hop_count": hop,
            "question_type": qtype,
            "answerable": ans,
            "planned_route": route_str,
            "structural_path": {
                "neo4j_edges_retrieved": len(neo4j_edges),
                "gold_relations_count": len(gold_rels),
                "recovered_relations_count": recovered_rels_count,
                "path_recovered": path_recovered,
                "false_positive_path": false_positive_path,
            },
            "chunk_retrieval": {
                "top_chunk_ids": h_res[:5],
                "gold_chunk_ids": gold_cids,
                "hit_at_1": hit1,
                "hit_at_3": hit3,
                "hit_at_5": hit5,
            },
            "generation": {
                "generated_answer": gen_answer,
                "gold_answer": gold_ans,
                "is_refusal": is_refusal,
                "answer_f1": round(ans_f1, 4),
                "keyword_recall": round(kw_recall, 4),
                "similarity": round(similarity, 4),
                "answer_correct": ans_correct,
            },
        })

    # Overall chunk metrics across answerable questions
    m_bm25 = _compute_metrics(bm25_runs, gold_chunk_runs)
    m_graph = _compute_metrics(graph_runs, gold_chunk_runs)
    m_hybrid = _compute_metrics(hybrid_runs, gold_chunk_runs)

    # Compute aggregate summaries
    total_ans = len(benchmark) - neg_stats["total"]
    total_path_hits = sum(hop_stats[h]["path_hits"] for h in [1, 2, 3])
    total_hit5_hits = sum(hop_stats[h]["hit5_hits"] for h in [1, 2, 3])
    total_ans_correct = sum(hop_stats[h]["ans_correct"] for h in [1, 2, 3])

    summary = {
        "total_questions": len(benchmark),
        "answerable_questions": total_ans,
        "negative_questions": neg_stats["total"],
        "structural_path_recovery": {
            "overall_rate": round(total_path_hits / total_ans, 4) if total_ans else 0.0,
            "hits": total_path_hits,
            "total": total_ans,
            "1_hop": {
                "total": hop_stats[1]["total"],
                "hits": hop_stats[1]["path_hits"],
                "rate": round(hop_stats[1]["path_hits"] / hop_stats[1]["total"], 4) if hop_stats[1]["total"] else 0.0,
            },
            "2_hop": {
                "total": hop_stats[2]["total"],
                "hits": hop_stats[2]["path_hits"],
                "rate": round(hop_stats[2]["path_hits"] / hop_stats[2]["total"], 4) if hop_stats[2]["total"] else 0.0,
            },
            "3_hop": {
                "total": hop_stats[3]["total"],
                "hits": hop_stats[3]["path_hits"],
                "rate": round(hop_stats[3]["path_hits"] / hop_stats[3]["total"], 4) if hop_stats[3]["total"] else 0.0,
            },
        },
        "negative_resistance": {
            "total_negative": neg_stats["total"],
            "correct_refusals": neg_stats["correct_refusals"],
            "refusal_accuracy": round(neg_stats["correct_refusals"] / neg_stats["total"], 4) if neg_stats["total"] else 0.0,
            "false_positive_rate": round(neg_stats["false_positive_paths"] / neg_stats["total"], 4) if neg_stats["total"] else 0.0,
            "false_positive_paths": neg_stats["false_positive_paths"],
            "hallucinated_answers": neg_stats["hallucinated_answers"],
        },
        "downstream_hit5_by_hop": {
            f"{h}_hop": {"hits": hop_stats[h]["hit5_hits"], "total": hop_stats[h]["total"]}
            for h in (1, 2, 3)
        },
        "downstream_chunk_retrieval": {
            "bm25": m_bm25,
            "graph": m_graph,
            "hybrid": m_hybrid,
        },
        "answer_correctness": {
            "answerable_correct": total_ans_correct,
            "answerable_total": total_ans,
            "answerable_accuracy": round(total_ans_correct / total_ans, 4) if total_ans else 0.0,
            "negative_refusal_accuracy": round(neg_stats["correct_refusals"] / neg_stats["total"], 4) if neg_stats["total"] else 0.0,
            "overall_accuracy": round((total_ans_correct + neg_stats["correct_refusals"]) / len(benchmark), 4),
            "by_hop": {
                "1_hop": {
                    "total": hop_stats[1]["total"],
                    "correct": hop_stats[1]["ans_correct"],
                    "accuracy": round(hop_stats[1]["ans_correct"] / hop_stats[1]["total"], 4) if hop_stats[1]["total"] else 0.0,
                    "mean_f1": round(sum(hop_stats[1]["f1_scores"]) / len(hop_stats[1]["f1_scores"]), 4) if hop_stats[1]["f1_scores"] else 0.0,
                },
                "2_hop": {
                    "total": hop_stats[2]["total"],
                    "correct": hop_stats[2]["ans_correct"],
                    "accuracy": round(hop_stats[2]["ans_correct"] / hop_stats[2]["total"], 4) if hop_stats[2]["total"] else 0.0,
                    "mean_f1": round(sum(hop_stats[2]["f1_scores"]) / len(hop_stats[2]["f1_scores"]), 4) if hop_stats[2]["f1_scores"] else 0.0,
                },
                "3_hop": {
                    "total": hop_stats[3]["total"],
                    "correct": hop_stats[3]["ans_correct"],
                    "accuracy": round(hop_stats[3]["ans_correct"] / hop_stats[3]["total"], 4) if hop_stats[3]["total"] else 0.0,
                    "mean_f1": round(sum(hop_stats[3]["f1_scores"]) / len(hop_stats[3]["f1_scores"]), 4) if hop_stats[3]["f1_scores"] else 0.0,
                },
            },
        },
    }

    return {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark_file": str(bench_file),
        "summary": summary,
        "detailed_results": eval_items,
    }


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d * 100:.1f}%)" if d else "n/a"


def _hit5_by_hop(data: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Per-hop downstream hybrid Hit@5 counts.

    Taken from the summary when present; recomputed from detailed_results for
    result files written before the summary carried them.
    """
    stored = data.get("summary", {}).get("downstream_hit5_by_hop")
    if stored:
        return stored
    out = {f"{h}_hop": {"hits": 0, "total": 0} for h in (1, 2, 3)}
    for item in data.get("detailed_results", []):
        if not item.get("answerable"):
            continue
        key = f"{item.get('hop_count')}_hop"
        if key in out:
            out[key]["total"] += 1
            out[key]["hits"] += int(bool(item.get("chunk_retrieval", {}).get("hit_at_5")))
    return out


def generate_markdown_report(data: Dict[str, Any]) -> str:
    """Render the report. Every number is computed from ``data``; nothing is hard-coded."""
    s = data["summary"]
    sp = s["structural_path_recovery"]
    nr = s["negative_resistance"]
    cr = s["downstream_chunk_retrieval"]
    ac = s["answer_correctness"]
    hit5 = _hit5_by_hop(data)
    details = data.get("detailed_results", [])
    negatives = [d for d in details if not d.get("answerable")]
    keyword_only = sum(
        1 for d in details
        if d.get("answerable")
        and d.get("generation", {}).get("answer_correct")
        and d.get("generation", {}).get("answer_f1", 0.0) < 0.25
    )
    n_hop = {h: sp[f"{h}_hop"]["total"] for h in (1, 2, 3)}
    lenient_total = ac["answerable_correct"] + nr["correct_refusals"]

    def delta(metric: str) -> str:
        return f"{(cr['hybrid'][metric] - cr['bm25'][metric]) * 100:+.1f} pts"

    def hop_row(h: int, label: str) -> str:
        key = f"{h}_hop"
        return (
            f"| {label} | {sp[key]['total']} | {_pct(sp[key]['hits'], sp[key]['total'])} "
            f"| {_pct(hit5[key]['hits'], hit5[key]['total'])} "
            f"| {_pct(ac['by_hop'][key]['correct'], ac['by_hop'][key]['total'])} "
            f"| {ac['by_hop'][key]['mean_f1']:.3f} |"
        )

    def mode_row(label: str, m: Dict[str, float]) -> str:
        return (
            f"| {label} | {m['Hit@1']:.3f} | {m['Hit@3']:.3f} | {m['Hit@5']:.3f} "
            f"| {m['Recall@5']:.3f} | {m['MRR']:.3f} |"
        )

    md = [
        "# LectureMIND — Held-Out GraphRAG Evaluation Report",
        "",
        f"**Dataset:** `{data['benchmark_file']}`  ",
        f"**Run timestamp:** {data.get('evaluation_timestamp', 'unknown')}  ",
        f"**Sample size:** {s['total_questions']} questions ({n_hop[1]} 1-hop, {n_hop[2]} 2-hop, "
        f"{n_hop[3]} 3-hop, {nr['total_negative']} negative)  ",
        "**Question / label provenance:** see `evaluation/outputs/graph_heldout_creation_metadata.json` "
        "and `evaluation/outputs/graph_heldout_manual_validation.json`.  ",
        "",
        "---",
        "",
        "## 1. Summary",
        "",
        "| Metric | Result | How it is measured |",
        "|---|---:|---|",
        f"| Structural path recovery (answerable) | {_pct(sp['hits'], sp['total'])} "
        "| Every gold relation of the question appears among the retrieved Neo4j edges "
        "(entity-name match, direction ignored) |",
        f"| Hybrid chunk Hit@5 (answerable) | {cr['hybrid']['Hit@5'] * 100:.1f}% "
        f"| Graph + BM25 fused by RRF. Versus BM25 alone: Hit@5 {delta('Hit@5')}, "
        f"Hit@1 {delta('Hit@1')}, MRR {delta('MRR')} |",
        f"| Negative refusals | {_pct(nr['correct_refusals'], nr['total_negative'])} "
        "| Generated answer contains a refusal phrase |",
        f"| False-positive path rate (negatives) | {_pct(nr['false_positive_paths'], nr['total_negative'])} "
        "| Graph returned an edge directly linking two trap entities |",
        f"| Lenient answer match (overall) | {_pct(lenient_total, s['total_questions'])} "
        "| Answerable: not a refusal AND (gold-entity keyword recall ≥ 0.5 OR token F1 ≥ 0.25). "
        f"{keyword_only} of {ac['answerable_correct']} answerable matches passed on keyword recall "
        "alone (F1 < 0.25). This is a lexical proxy, not a correctness judgment. |",
        "",
        "---",
        "",
        "## 2. Hop-by-Hop Breakdown",
        "",
        "| Category | N | Path recovery | Hybrid Hit@5 | Lenient answer match | Mean token F1 |",
        "|---|---:|---:|---:|---:|---:|",
        hop_row(1, "1-hop"),
        hop_row(2, "2-hop"),
        hop_row(3, "3-hop"),
        f"| Negative | {nr['total_negative']} | {nr['false_positive_paths']} false-positive path(s) "
        f"| n/a | {_pct(nr['correct_refusals'], nr['total_negative'])} refused | n/a |",
        "",
        "---",
        "",
        "## 3. Evidence Chunk Retrieval (answerable questions)",
        "",
        "| Mode | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR (full list) |",
        "|---|---:|---:|---:|---:|---:|",
        mode_row("BM25 lexical", cr["bm25"]),
        mode_row("Graph-only", cr["graph"]),
        mode_row("Hybrid (Graph + BM25, RRF)", cr["hybrid"]),
        "",
        "---",
        "",
        "## 4. Negative Questions",
        "",
    ]
    if negatives:
        for d in negatives:
            fp = d.get("structural_path", {}).get("false_positive_path")
            refused = d.get("generation", {}).get("answer_correct")
            md.append(
                f"- **{d.get('question_id', '?')}:** {d.get('question', '')[:120]} — "
                f"graph linked trap entities: {'yes' if fp else 'no'}; "
                f"answer refused: {'yes' if refused else 'no'}"
            )
    else:
        md.append("_No per-question details in this result file._")
    md += [
        "",
        "---",
        "",
        "## 5. Scope and Caveats",
        "",
        "1. **Not the production pipeline.** Chunk retrieval here uses `SimpleBM25` + "
        "`SimpleGraphRetriever` fused by RRF, and the prompt context is the top-3 chunks plus "
        "up to 5 graph edges. Dense retrieval, cross-encoder reranking and `ContextBuilder` are "
        "not exercised, so these numbers do not measure the deployed query workflow.",
        f"2. **Small sample.** {n_hop[1]}/{n_hop[2]}/{n_hop[3]} answerable questions per hop "
        f"and {nr['total_negative']} negatives; one question moves the smallest per-hop rate by "
        f"{100 / min((v for v in n_hop.values() if v), default=1):.0f} points.",
        "3. **Lexical answer scoring.** \"Lenient answer match\" is keyword/F1 based; gold entity "
        "names usually appear in the question itself. Use a calibrated judge or human review "
        "for answer correctness.",
        "4. **Cached generations.** Answers are reused from "
        "`evaluation/outputs/graph_heldout_checkpoint.json` when present; delete it to regenerate "
        "answers after pipeline changes.",
    ]
    return "\n".join(md)


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Held-out GraphRAG evaluation")
    parser.add_argument(
        "--from-json",
        help="Regenerate the Markdown report from an existing result JSON "
             "(no Neo4j / LLM calls, results unchanged).",
    )
    args = parser.parse_args(argv)

    out_dir = Path("evaluation/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "graph_heldout_evaluation.md"

    if args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as f:
            results = json.load(f)
        md_path.write_text(generate_markdown_report(results), encoding="utf-8")
        print(f"Regenerated {md_path} from {args.from_json}")
        return

    results = run_evaluation()

    json_path = out_dir / "graph_heldout_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_path.write_text(generate_markdown_report(results), encoding="utf-8")

    print(f"Evaluation complete. Saved JSON to {json_path} and Markdown to {md_path}")
    summary = results["summary"]
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Structural Path Recovery: {summary['structural_path_recovery']['overall_rate']*100:.1f}% ({summary['structural_path_recovery']['hits']}/{summary['structural_path_recovery']['total']})")
    print(f"Hybrid Chunk Hit@5: {summary['downstream_chunk_retrieval']['hybrid']['Hit@5']*100:.1f}%")
    print(f"Negative Refusal Accuracy: {summary['negative_resistance']['refusal_accuracy']*100:.1f}%")
    print(f"Lenient Answer Match (overall): {summary['answer_correctness']['overall_accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()
