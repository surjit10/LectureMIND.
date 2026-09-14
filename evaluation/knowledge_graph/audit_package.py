# evaluation/knowledge_graph/audit_package.py
# Comprehensive Knowledge Graph Quality Audit Runner for LectureMIND.
#
# Reads a knowledge package ZIP, executes read-only structural & gold-standard audits,
# evaluates downstream GraphRAG retrieval, and generates honest, reproducible reports.
#
# Honesty guarantees (do not regress these):
#   1. Gold-standard metrics are reported as None when the packaged gold files do
#      not apply to the audited lecture — they are NEVER substituted with defaults.
#   2. The composite diagnostic score only sums components that were actually
#      measured. Unmeasurable components are omitted, and the score is reported
#      as "<score> / <max>" rather than a false "x / 100".
#   3. The Markdown report is generated from measured values and the package
#      manifest. No lecture-specific narrative is hard-coded in the template.
#   4. QA benchmarks whose expected_chunk_ids do not belong to the audited
#      package are reported as N/A instead of silently scoring 0.0.

import argparse
import json
import logging
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud.extraction.prerequisite_inference import infer_prerequisites
from evaluation.knowledge_graph.entity_auditor import audit_entities
from evaluation.knowledge_graph.graphrag_evaluator import evaluate_graphrag
from evaluation.knowledge_graph.ontology_analyzer import analyze_ontology
from evaluation.knowledge_graph.prerequisite_auditor import audit_prerequisites
from evaluation.knowledge_graph.relation_auditor import audit_relations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("kg_audit_runner")


# Composite-score component weights. Only components that are actually
# measurable contribute; the denominator is the sum of contributing weights.
_SCORE_WEIGHTS = {
    "entity_quality": 20.0,
    "relation_quality": 20.0,
    "direction_accuracy": 10.0,
    "evidence_grounding": 15.0,
    "prerequisite_dag": 15.0,
    "graph_coherence": 10.0,
    "graphrag_usefulness": 10.0,
}


def _resolve_gold_lecture_id(
    gold_entities_path: Optional[str | Path],
    gold_relations_path: Optional[str | Path],
    gold_prerequisites_path: Optional[str | Path],
) -> Optional[str]:
    """Return the lecture_id the gold files were annotated for, if discoverable."""
    for p in (gold_entities_path, gold_relations_path, gold_prerequisites_path):
        if p and Path(p).exists():
            try:
                with open(p, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                lid = (data.get("metadata") or {}).get("lecture_id")
                if lid:
                    return lid
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _resolve_benchmark_lecture_id(benchmark_data: List[Dict[str, Any]]) -> Optional[str]:
    """Infer the lecture a QA benchmark was written for from its chunk-ID prefixes."""
    prefixes = set()
    for sample in benchmark_data:
        for cid in sample.get("expected_chunk_ids", []) or []:
            # Chunk IDs conventionally look like "<lecture_id>_chunk_NNNNNN".
            prefix = cid.rsplit("_chunk_", 1)[0]
            if prefix and prefix != cid:
                prefixes.add(prefix)
    if len(prefixes) == 1:
        return next(iter(prefixes))
    if len(prefixes) > 1:
        return f"<multiple:{sorted(prefixes)[:2]}...>"
    return None


def run_full_audit(
    package_zip_path: str | Path,
    output_dir: str | Path = "outputs/kg_quality",
    gold_entities_path: Optional[str | Path] = "evaluation/knowledge_graph/entity_gold.json",
    gold_relations_path: Optional[str | Path] = "evaluation/knowledge_graph/relation_gold.json",
    gold_prerequisites_path: Optional[str | Path] = "evaluation/knowledge_graph/prerequisite_gold.json",
    benchmark_dataset_path: Optional[str | Path] = "evaluation/datasets/transformer_kg_qa_50.json",
) -> Dict[str, Any]:
    """
    Execute full knowledge graph audit on a knowledge package ZIP.
    """
    pkg_path = Path(package_zip_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Opening knowledge package: %s", pkg_path)
    with zipfile.ZipFile(pkg_path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        entities = json.loads(zf.read("entities.json").decode("utf-8"))
        relations = json.loads(zf.read("relations.json").decode("utf-8"))
        chunks = json.loads(zf.read("multimodal_chunks.json").decode("utf-8"))
        segments = json.loads(zf.read("segments.json").decode("utf-8"))
        metadata = json.loads(zf.read("metadata.json").decode("utf-8")) if "metadata.json" in zf.namelist() else {}

    lecture_id = manifest.get("lecture_id", pkg_path.stem)
    total_chunks = len(chunks)
    total_entities = len(entities)
    total_relations = len(relations)
    total_segments = len(segments)
    duration_s = float(metadata.get("duration") or manifest.get("duration") or 0.0)
    logger.info(
        "Auditing lecture '%s' (chunks=%d, entities=%d, relations=%d)",
        lecture_id, total_chunks, total_entities, total_relations,
    )

    # ── Gold applicability: gold files are annotated per-lecture. Auditing a
    #    different lecture against them would produce meaningless 0.0 F1s
    #    (verified: all-zero gold metrics for CS162/MIT/Self-Attention).
    gold_lecture_id = _resolve_gold_lecture_id(gold_entities_path, gold_relations_path, gold_prerequisites_path)
    gold_applies = (
        gold_lecture_id is not None
        and (gold_lecture_id in lecture_id or lecture_id in gold_lecture_id)
    )
    if not gold_applies:
        logger.warning(
            "Gold reference files are annotated for lecture '%s' but the audited "
            "package is '%s'. Gold P/R/F1 will be reported as null (N/A) instead "
            "of misleading zeros.",
            gold_lecture_id, lecture_id,
        )
        gold_entities_path = None
        gold_relations_path = None
        gold_prerequisites_path = None

    # 1. Entity Audit
    logger.info("Running Entity Quality Audit...")
    entity_result = audit_entities(
        entities=entities,
        relations=relations,
        gold_entities_path=gold_entities_path,
    )

    # 2. Relation & Direction Audit
    logger.info("Running Relation & Direction Audit...")
    relation_result = audit_relations(
        relations=relations,
        entities=entities,
        chunks=chunks,
        gold_relations_path=gold_relations_path,
    )

    # 3. Prerequisite Inference & DAG Audit
    logger.info("Running Prerequisite Inference & DAG Audit...")
    inferred_data = infer_prerequisites(
        entities=entities,
        chunks=chunks,
        segments=segments,
        relations=relations,
    )
    inferred_prereqs = inferred_data.get("prerequisites", [])

    prereq_result = audit_prerequisites(
        prerequisites=inferred_prereqs,
        entities=entities,
        chunks=chunks,
        gold_prerequisites_path=gold_prerequisites_path,
    )

    # 4. Ontology Diagnostic Analysis
    logger.info("Running Ontology Diagnostic Analysis...")
    ontology_result = analyze_ontology(relations, entities)

    # 5. GraphRAG Downstream Evaluation
    logger.info("Running GraphRAG Downstream Evaluation...")
    benchmark_data: List[Dict[str, Any]] = []
    benchmark_applies = False
    benchmark_lecture_id: Optional[str] = None
    if benchmark_dataset_path and Path(benchmark_dataset_path).exists():
        with open(benchmark_dataset_path, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)
        benchmark_lecture_id = _resolve_benchmark_lecture_id(benchmark_data)
        benchmark_applies = bool(benchmark_lecture_id) and (
            benchmark_lecture_id in lecture_id or lecture_id in benchmark_lecture_id
        )
        if benchmark_data and not benchmark_applies:
            logger.warning(
                "QA benchmark expected_chunk_ids belong to lecture '%s' but the "
                "audited package is '%s'. Downstream GraphRAG metrics will be "
                "reported as null (N/A) instead of misleading zeros.",
                benchmark_lecture_id, lecture_id,
            )

    graphrag_result = None
    if benchmark_data and benchmark_applies:
        graphrag_result = evaluate_graphrag(
            chunks=chunks,
            entities=entities,
            relations=relations,
            benchmark_dataset=benchmark_data,
        )

    # 6. Composite Diagnostic Score — only measurable components contribute.
    #    Falsy-zero metrics are NEVER replaced with optimistic defaults.
    score_breakdown: Dict[str, float] = {}
    if gold_applies:
        score_breakdown["entity_quality"] = round(
            (entity_result.gold_entity_f1 if entity_result.gold_entity_f1 is not None
             else entity_result.heuristic_accept_rate) * _SCORE_WEIGHTS["entity_quality"], 2)
        score_breakdown["relation_quality"] = round(
            (relation_result.gold_relation_f1 or 0.0) * _SCORE_WEIGHTS["relation_quality"], 2)
        score_breakdown["direction_accuracy"] = round(
            (relation_result.direction_accuracy or 0.0) * _SCORE_WEIGHTS["direction_accuracy"], 2)
    else:
        score_breakdown["entity_quality"] = round(
            entity_result.heuristic_accept_rate * _SCORE_WEIGHTS["entity_quality"], 2)
    score_breakdown["evidence_grounding"] = round(
        relation_result.evidence_coverage_rate * _SCORE_WEIGHTS["evidence_grounding"], 2)
    # Graph coherence rewards DEPTH and CONNECTIVITY, not structural spread:
    #   - direct-evidence rate: share of relations backed by verbatim lecture text
    #     (the strongest grounding tier; indirect/co-occurrence are scored via
    #     evidence_grounding above)
    #   - entity participation: 1 - orphan_rate (share of entities that the graph
    #     actually connects into at least one relation)
    # The raw cross-chunk ratio is still reported descriptively (Section 5 of the
    # report) but is no longer scored: a high ratio historically rewarded sparse
    # graphs whose few relations happened to span chunks, and penalized dense,
    # well-grounded same-chunk relations.
    direct_evidence_rate = (
        relation_result.direct_evidence_count / relation_result.total_relations
        if relation_result.total_relations > 0 else 0.0
    )
    entity_participation = 1.0 - entity_result.orphan_rate
    graph_coherence = 0.5 * (direct_evidence_rate + entity_participation)
    score_breakdown["graph_coherence"] = round(
        graph_coherence * _SCORE_WEIGHTS["graph_coherence"], 2)
    # The prerequisite component is gold-dependent (gold F1 when the gold labels apply to this
    # lecture, fuzzy self-F1 as a diagnostic otherwise). When no reference labels apply at all
    # (prereq F1 is None) the component is UNMEASURABLE, so it is excluded from both the score
    # and the denominator — DAG-ness itself is reported structurally, never scored as 0.
    prereq_f1 = prereq_result.gold_prerequisite_f1 if gold_applies else None
    if prereq_f1 is None:
        prereq_f1 = prereq_result.fuzzy_prerequisite_f1
    if prereq_f1 is not None:
        score_breakdown["prerequisite_dag"] = round(
            (_SCORE_WEIGHTS["prerequisite_dag"] if prereq_result.is_dag else _SCORE_WEIGHTS["prerequisite_dag"] / 3) * prereq_f1, 2)
    if graphrag_result is not None:
        score_breakdown["graphrag_usefulness"] = round(
            graphrag_result.hybrid_rag_metrics["Recall@5"] * _SCORE_WEIGHTS["graphrag_usefulness"], 2)

    composite_diagnostic_score = round(sum(score_breakdown.values()), 1)
    composite_max = round(sum(_SCORE_WEIGHTS[k] for k in score_breakdown), 1)
    composite_display = f"{composite_diagnostic_score} / {composite_max}"

    # 7. Serialize Artifacts
    # A. entity_audit.json
    (out_dir / "entity_audit.json").write_text(json.dumps(asdict(entity_result), indent=2), encoding="utf-8")

    # B. relation_audit.json
    (out_dir / "relation_audit.json").write_text(json.dumps(asdict(relation_result), indent=2), encoding="utf-8")

    # C. prerequisite_audit.json
    (out_dir / "prerequisite_audit.json").write_text(json.dumps(asdict(prereq_result), indent=2), encoding="utf-8")

    # D. evidence_audit.json
    evidence_payload = {
        "direct_evidence_count": relation_result.direct_evidence_count,
        "indirect_evidence_count": relation_result.indirect_evidence_count,
        "co_occurrence_only_count": relation_result.co_occurrence_only_count,
        "no_evidence_count": relation_result.no_evidence_count,
        "evidence_coverage_rate": relation_result.evidence_coverage_rate,
        "relations_with_evidence": [
            r for r in relation_result.relation_details if r["evidence_tier"] in ("DIRECT_EVIDENCE", "INDIRECT_EVIDENCE")
        ],
        "relations_lacking_evidence": [
            r for r in relation_result.relation_details if r["evidence_tier"] in ("CO_OCCURRENCE_ONLY", "NO_EVIDENCE")
        ],
    }
    (out_dir / "evidence_audit.json").write_text(json.dumps(evidence_payload, indent=2), encoding="utf-8")

    # E. metrics.json — null (None) means "not measurable for this package".
    metrics_payload = {
        "lecture_id": lecture_id,
        "composite_diagnostic_score": composite_diagnostic_score,
        "composite_max": composite_max,
        "gold_applies": gold_applies,
        "gold_lecture_id": gold_lecture_id,
        "benchmark_applies": benchmark_applies,
        "benchmark_lecture_id": benchmark_lecture_id,
        "score_breakdown": score_breakdown,
        "entity_metrics": {
            "total_entities": entity_result.total_entities,
            "fragment_rate": entity_result.fragment_rate,
            "orphan_rate": entity_result.orphan_rate,
            "gold_precision": entity_result.gold_entity_precision,
            "gold_recall": entity_result.gold_entity_recall,
            "gold_f1": entity_result.gold_entity_f1,
        },
        "relation_metrics": {
            "total_relations": relation_result.total_relations,
            "evidence_coverage_rate": relation_result.evidence_coverage_rate,
            "cross_chunk_ratio": relation_result.cross_chunk_ratio,
            "gold_precision": relation_result.gold_relation_precision,
            "gold_recall": relation_result.gold_relation_recall,
            "gold_f1": relation_result.gold_relation_f1,
            "direction_accuracy": relation_result.direction_accuracy,
        },
        "prerequisite_metrics": {
            "total_prerequisites": prereq_result.total_prerequisites,
            "is_dag": prereq_result.is_dag,
            "cycle_count": prereq_result.cycle_count,
            "gold_precision": prereq_result.gold_prerequisite_precision,
            "gold_recall": prereq_result.gold_prerequisite_recall,
            "gold_f1": prereq_result.gold_prerequisite_f1,
        },
        "graphrag_metrics": {
            "bm25": graphrag_result.bm25_metrics if graphrag_result else None,
            "vector_rag": graphrag_result.bm25_metrics if graphrag_result else None,
            "graph_rag": graphrag_result.graph_rag_metrics if graphrag_result else None,
            "hybrid_rag": graphrag_result.hybrid_rag_metrics if graphrag_result else None,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # F. Markdown Quality Report (all values measured, none hard-coded)
    report_md = _generate_markdown_report(
        lecture_id=lecture_id,
        manifest=manifest,
        total_chunks=total_chunks,
        total_segments=total_segments,
        duration_s=duration_s,
        entity_res=entity_result,
        rel_res=relation_result,
        prereq_res=prereq_result,
        ontology_res=ontology_result,
        rag_res=graphrag_result,
        benchmark_applies=benchmark_applies,
        benchmark_path=str(benchmark_dataset_path) if benchmark_dataset_path else None,
        gold_applies=gold_applies,
        composite_score=composite_diagnostic_score,
        composite_max=composite_max,
        score_breakdown=score_breakdown,
    )
    (out_dir / "knowledge_graph_quality_report.md").write_text(report_md, encoding="utf-8")

    logger.info("Audit complete! Reports and metrics written to %s", out_dir)
    return metrics_payload


def _fmt_pct(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def _fmt_val(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


def _generate_markdown_report(
    lecture_id: str,
    manifest: Dict[str, Any],
    total_chunks: int,
    total_segments: int,
    duration_s: float,
    entity_res: Any,
    rel_res: Any,
    prereq_res: Any,
    ontology_res: Any,
    rag_res: Any,
    benchmark_applies: bool,
    benchmark_path: Optional[str],
    gold_applies: bool,
    composite_score: float,
    composite_max: float,
    score_breakdown: Dict[str, float],
) -> str:
    """Generate a measured-values-only Markdown report (no hard-coded narrative)."""

    accepted = entity_res.total_entities - entity_res.fragment_count
    tp = rel_res.tp_relation_count
    total_rel = max(rel_res.total_relations, 1)

    md = f"""# LectureMIND Knowledge Graph Quality Report

**Lecture ID**: `{lecture_id}`
**Evaluation Standard**: Read-Only Structural Audit & Reference-Label Evaluation
**KG Quality Diagnostic Score**: **{composite_score} / {composite_max}** *(only measurable components counted)*

> [!NOTE]
> **Reference Label Provenance**: Reference concepts, relations, and prerequisite dependencies are **LLM-assisted reference labels, pending human verification**. They serve as an automated evaluation benchmark and have not undergone independent manual double-blind verification by human domain experts.

---

## 1. Dataset & Pipeline Summary

| Metric | Measured Value |
| :--- | :--- |
| **Lecture Duration** | {duration_s:.1f}s (~{duration_s / 60:.1f} min) |
| **Multimodal Chunks** | {total_chunks} chunks |
| **Segments** | {total_segments} segments |
| **Extracted Entities** | {entity_res.total_entities} (accepted: {accepted}, flagged as fragments: {entity_res.fragment_count}) |
| **Extracted Relations** | {rel_res.total_relations} (dangling entity references: {rel_res.dangling_entity_count}) |
| **Inferred Prerequisites** | {prereq_res.total_prerequisites} edges (DAG: {prereq_res.is_dag}, cycles: {prereq_res.cycle_count}) |
| **QA Benchmark Applied** | {"Yes — " + benchmark_path if benchmark_applies else "No — benchmark chunk IDs do not belong to this package (reported N/A)"} |
| **Gold Reference Applied** | {"Yes" if gold_applies else "No — gold labels are annotated for a different lecture (reported N/A)"} |

---

## 2. Entity Extraction Quality
"""
    if gold_applies and entity_res.gold_entity_f1 is not None:
        md += f"""
* **Reference Label Standard**: `{entity_res.gold_evaluation_method}`
* **Reference Concepts Available**: {entity_res.gold_entity_count}

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Reference Entity Precision** | **{_fmt_pct(entity_res.gold_entity_precision)}** | {entity_res.tp_entity_count} / {entity_res.total_entities} extracted entities match a reference concept (1-to-1) |
| **Reference Entity Recall** | **{_fmt_pct(entity_res.gold_entity_recall)}** | {entity_res.matched_gold_entity_count} / {entity_res.gold_entity_count} reference concepts captured |
| **Reference Entity F1** | **{_fmt_pct(entity_res.gold_entity_f1)}** | Balanced entity extraction performance |
"""
    else:
        md += """
* Gold reference metrics: **N/A** — the packaged gold labels are annotated for a different lecture and were NOT applied.
"""
    md += f"""
| Structural Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Fragment Rate** | **{_fmt_pct(entity_res.fragment_rate)}** | {entity_res.fragment_count} sentence-fragment extraction artifacts detected |
| **Generic Non-Concept Rate** | **{_fmt_pct(entity_res.generic_rate)}** | {entity_res.generic_count} generic stopwords detected |
| **Orphan Entity Rate** | **{_fmt_pct(entity_res.orphan_rate)}** | {entity_res.orphan_count} entities with graph degree = 0 |
| **Duplicate Surface Forms** | **{entity_res.duplicate_groups_count}** | groups of entities sharing a normalized surface form |

### Flagged Entity Issues:
"""
    for issue in entity_res.issues:
        md += f"- **[{issue['issue_type']}]** `{issue['entity_name']}`: {issue['reason']}\n"

    md += f"""
---

## 3. Relation Extraction, Direction & Evidence Grounding Quality
"""
    if gold_applies and rel_res.gold_relation_f1 is not None:
        matched_pairs_total = rel_res.correct_direction_count + rel_res.reversed_direction_count
        md += f"""
* **Reference Label Standard**: `{rel_res.gold_evaluation_method}` ({rel_res.gold_relation_count} reference relations)

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Reference Relation Precision** | **{_fmt_pct(rel_res.gold_relation_precision)}** | {tp} / {rel_res.total_relations} extracted relations match a reference triple |
| **Reference Relation Recall** | **{_fmt_pct(rel_res.gold_relation_recall)}** | {rel_res.matched_gold_relation_count} / {rel_res.gold_relation_count} reference triples captured |
| **Reference Relation F1** | **{_fmt_pct(rel_res.gold_relation_f1)}** | Harmonic mean of precision and recall |
| **Gold-Matched Direction Accuracy** | **{_fmt_pct(rel_res.gold_matched_direction_accuracy)}** | {rel_res.correct_direction_count} / {matched_pairs_total} gold-matched directional relations have correct orientation |
| **Evidence Coverage Rate** | **{_fmt_pct(rel_res.evidence_coverage_rate)}** | {rel_res.direct_evidence_count + rel_res.indirect_evidence_count} / {rel_res.total_relations} relations supported by direct or indirect transcript evidence |
"""
    else:
        md += """
* Gold reference metrics: **N/A** — the packaged gold labels are annotated for a different lecture and were NOT applied.
"""
    md += f"""
| Evidence Grounding Tier | Count | Proportion |
| :--- | :---: | :---: |
| **Direct Textual Evidence** | {rel_res.direct_evidence_count} | {rel_res.direct_evidence_count / total_rel * 100:.1f}% |
| **Indirect Textual Evidence** | {rel_res.indirect_evidence_count} | {rel_res.indirect_evidence_count / total_rel * 100:.1f}% |
| **Co-occurrence Only** *(not evidence)* | {rel_res.co_occurrence_only_count} | {rel_res.co_occurrence_only_count / total_rel * 100:.1f}% |
| **No Single-Chunk Evidence** | {rel_res.no_evidence_count} | {rel_res.no_evidence_count / total_rel * 100:.1f}% |

### Flagged Directional Warnings:
"""
    directional = [i for i in rel_res.issues if i["issue_type"] == "SUSPICIOUS_DIRECTION"]
    if directional:
        for issue in directional:
            md += f"- **[{issue['issue_type']}]** `{issue['source_name']} -[{issue['relation']}]-> {issue['target_name']}`: {issue['reason']}\n"
    else:
        md += "- None detected.\n"

    md += f"""
---

## 4. Prerequisite Dependency Quality (DAG Validation)

| Prerequisite Metric | Value | Status |
| :--- | :---: | :---: |
| **Total Inferred Prerequisites** | {prereq_res.total_prerequisites} | Multi-signal inferred edges |
| **Graph Topology (Is DAG)** | **{prereq_res.is_dag}** | {'Strict DAG — zero cycles' if prereq_res.is_dag else 'Cycles detected'} |
| **Cycle Count** | **{prereq_res.cycle_count}** | {'PASS' if prereq_res.cycle_count == 0 else 'FAIL'} |
| **Self-Loop Count** | **{prereq_res.self_loop_count}** | {'PASS' if prereq_res.self_loop_count == 0 else 'FAIL'} |
| **Temporal Inversion Warnings** | **{prereq_res.temporal_inversion_count}** | Diagnostic (LOW severity) |
| **Non-Pedagogical Flags** | **{prereq_res.total_prerequisites - prereq_res.pedagogical_count}** | Marker-based heuristic |
"""
    if gold_applies and prereq_res.gold_prerequisite_f1 is not None:
        md += f"""
| **Reference Prerequisite Precision** | **{_fmt_pct(prereq_res.gold_prerequisite_precision)}** | {prereq_res.tp_prerequisite_count} / {prereq_res.total_prerequisites} inferred prerequisites match reference DAG (strict 1-to-1) |
| **Reference Prerequisite Recall** | **{_fmt_pct(prereq_res.gold_prerequisite_recall)}** | {prereq_res.matched_gold_prerequisite_count} / {prereq_res.gold_prerequisite_count} reference prerequisites captured |
| **Reference Prerequisite F1** | **{_fmt_pct(prereq_res.gold_prerequisite_f1)}** | Prerequisite graph alignment score |
"""
    else:
        md += """
* Gold reference metrics: **N/A** — the packaged gold prerequisite labels are annotated for a different lecture and were NOT applied.
"""

    md += f"""
---

## 5. Cross-Chunk Relation Analysis (descriptive — not scored)

The cross-chunk ratio is reported for descriptive purposes only: it reflects how far
relations span the lecture timeline, and correlates with package density — sparse graphs
mechanically show a higher share. It is **not** a quality signal and is excluded from the
composite score, which instead rewards grounding depth and entity participation (see
`graph_coherence` in Section 8).

| Metric | Measured Value |
| :--- | :--- |
| **Same-Chunk Relations** | {rel_res.same_chunk_count} ({100 - rel_res.cross_chunk_ratio * 100:.1f}%) |
| **Cross-Chunk Relations** | **{rel_res.cross_chunk_count} ({rel_res.cross_chunk_ratio * 100:.1f}%)** |
| **Cross-Chunk Ratio** | **{rel_res.cross_chunk_ratio:.3f}** |

---

## 6. Downstream GraphRAG Benchmark Evaluation
"""
    if rag_res is not None:
        md += f"""
Evaluated on the applied QA benchmark ({rag_res.total_queries} queries; lexical BM25 baseline vs graph-only vs hybrid RRF — self-contained evaluators, not the production Qdrant/Neo4j stack):

| Retrieval Route | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Precision@5 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 / Lexical Retrieval** | **{rag_res.bm25_metrics['Hit@1']:.2f}** | **{rag_res.bm25_metrics['Hit@3']:.2f}** | **{rag_res.bm25_metrics['Hit@5']:.2f}** | **{rag_res.bm25_metrics['Recall@5']:.3f}** | **{rag_res.bm25_metrics['Precision@5']:.3f}** | **{rag_res.bm25_metrics['MRR']:.3f}** |
| **Graph-Only RAG** | {rag_res.graph_rag_metrics['Hit@1']:.2f} | {rag_res.graph_rag_metrics['Hit@3']:.2f} | {rag_res.graph_rag_metrics['Hit@5']:.2f} | {rag_res.graph_rag_metrics['Recall@5']:.3f} | {rag_res.graph_rag_metrics['Precision@5']:.3f} | {rag_res.graph_rag_metrics['MRR']:.3f} |
| **Hybrid RAG (RRF)** | {rag_res.hybrid_rag_metrics['Hit@1']:.2f} | {rag_res.hybrid_rag_metrics['Hit@3']:.2f} | **{rag_res.hybrid_rag_metrics['Hit@5']:.2f}** | {rag_res.hybrid_rag_metrics['Recall@5']:.3f} | {rag_res.hybrid_rag_metrics['Precision@5']:.3f} | {rag_res.hybrid_rag_metrics['MRR']:.3f} |

> [!IMPORTANT]
> **Methodological note:** BM25 is lexical retrieval (term frequency + length saturation), not dense vector search. Dense multimodal-embedding retrieval is a planned future experiment.
"""
    else:
        md += """
* **N/A** — the configured QA benchmark's `expected_chunk_ids` belong to a different lecture package. Scoring this package against them would only produce meaningless zeros, so downstream retrieval metrics are not reported. Run the audit with `--benchmark-qa` pointing at a benchmark written for this package.
"""

    md += f"""
---

## 7. Diagnostic Ontology Breakdown

* `USED_BY`: {ontology_res.used_by_breakdown['total_used_by']} relations ({ontology_res.used_by_breakdown['percentage_of_all_relations']}% of all relations)
"""
    for role, info in ontology_res.used_by_breakdown["roles"].items():
        md += f"  - **{role}**: {info['count']} instances — {info['description']}\n"
        if info["examples"]:
            md += f"    Examples: `{'`, `'.join(info['examples'])}`\n"

    md += f"""
---

## 8. Composite Score Breakdown (only measurable components)

| Component | Weight | Contribution |
| :--- | :---: | :---: |
"""
    for component, contribution in score_breakdown.items():
        md += f"| {component} | {_SCORE_WEIGHTS[component]:.0f} | {contribution} |\n"
    md += f"| **Total** | **{composite_max:.0f}** | **{composite_score}** |\n"

    md += """
---

## 9. Recommendations for Next Iterations

1. **[Medium Priority] Entity Hygiene Pre-Filter**: Filter clausal-fragment entities before saving Stage A8 output.
2. **[Medium Priority] Per-Lecture Gold Labels**: Produce human-verified gold files for every lecture that is claimed to be quality-audited; this audit reports N/A rather than fabricating scores when they are missing.
3. **[Low Priority] Ontology Expansion**: Introduce `COMPONENT_OF` where `USED_BY` is semantically overloaded.
"""
    return md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LectureMIND KG Quality Auditor")
    parser.add_argument("--package", required=True, help="Path to knowledge package ZIP")
    parser.add_argument("--output-dir", default="outputs/kg_quality", help="Output directory for reports")
    parser.add_argument("--gold-entities", default="evaluation/knowledge_graph/entity_gold.json")
    parser.add_argument("--gold-relations", default="evaluation/knowledge_graph/relation_gold.json")
    parser.add_argument("--gold-prerequisites", default="evaluation/knowledge_graph/prerequisite_gold.json")
    parser.add_argument("--benchmark-qa", default="evaluation/datasets/transformer_kg_qa_50.json")

    args = parser.parse_args()
    metrics = run_full_audit(
        package_zip_path=args.package,
        output_dir=args.output_dir,
        gold_entities_path=args.gold_entities,
        gold_relations_path=args.gold_relations,
        gold_prerequisites_path=args.gold_prerequisites,
        benchmark_dataset_path=args.benchmark_qa,
    )
    print("\n=== COMPOSITE DIAGNOSTIC SUMMARY ===")
    print(f"Composite Diagnostic Score: {metrics['composite_diagnostic_score']} / {metrics['composite_max']}")
    print(f"Gold applies:               {metrics['gold_applies']} (gold lecture: {metrics['gold_lecture_id']})")
    print(f"Benchmark applies:          {metrics['benchmark_applies']} (benchmark lecture: {metrics['benchmark_lecture_id']})")
    print(f"Gold Entity F1:             {metrics['entity_metrics']['gold_f1']}")
    print(f"Gold Relation F1:           {metrics['relation_metrics']['gold_f1']}")
    print(f"Gold Prerequisite F1:       {metrics['prerequisite_metrics']['gold_f1']}")
    print(f"Evidence Coverage Rate:     {metrics['relation_metrics']['evidence_coverage_rate']}")
    print(f"Cross Chunk Ratio:          {metrics['relation_metrics']['cross_chunk_ratio']}")
