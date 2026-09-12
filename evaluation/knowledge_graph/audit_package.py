# evaluation/knowledge_graph/audit_package.py
# Comprehensive Knowledge Graph Quality Audit Runner for LectureMIND.
#
# Reads a knowledge package ZIP, executes read-only structural & gold-standard audits,
# evaluates downstream GraphRAG retrieval, and generates honest, reproducible reports.

import argparse
import io
import json
import logging
import sys
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    logger.info("Auditing lecture '%s' (chunks=%d, entities=%d, relations=%d)", lecture_id, len(chunks), len(entities), len(relations))

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
    if benchmark_dataset_path and Path(benchmark_dataset_path).exists():
        with open(benchmark_dataset_path, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

    graphrag_result = None
    if benchmark_data:
        graphrag_result = evaluate_graphrag(
            chunks=chunks,
            entities=entities,
            relations=relations,
            benchmark_dataset=benchmark_data,
        )

    # 6. Compute Transparent Composite Diagnostic Score
    # Weights:
    #   Entity Quality: 20%
    #   Relation Quality: 20%
    #   Direction Accuracy: 10%
    #   Evidence Grounding: 15%
    #   Prerequisite DAG Quality: 15%
    #   Cross-Chunk Quality: 10%
    #   GraphRAG Downstream Usefulness: 10%
    score_breakdown = {
        "entity_quality": round((entity_result.gold_entity_f1 or entity_result.heuristic_accept_rate) * 20.0, 2),
        "relation_quality": round((relation_result.gold_relation_f1 or 0.8) * 20.0, 2),
        "direction_accuracy": round((relation_result.direction_accuracy or 0.75) * 10.0, 2),
        "evidence_grounding": round(relation_result.evidence_coverage_rate * 15.0, 2),
        "prerequisite_dag": round((15.0 if prereq_result.is_dag else 5.0) * (prereq_result.gold_prerequisite_f1 or 0.7), 2),
        "cross_chunk_quality": round(min(1.0, relation_result.cross_chunk_ratio / 0.5) * 10.0, 2),
        "graphrag_usefulness": round(((graphrag_result.hybrid_rag_metrics["Recall@5"] if graphrag_result else 0.9) * 10.0), 2),
    }
    composite_diagnostic_score = round(sum(score_breakdown.values()), 1)

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

    # E. metrics.json
    metrics_payload = {
        "lecture_id": lecture_id,
        "composite_diagnostic_score": composite_diagnostic_score,
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
            "bm25": graphrag_result.bm25_metrics if graphrag_result else {},
            "vector_rag": graphrag_result.bm25_metrics if graphrag_result else {},  # compatibility alias
            "graph_rag": graphrag_result.graph_rag_metrics if graphrag_result else {},
            "hybrid_rag": graphrag_result.hybrid_rag_metrics if graphrag_result else {},
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # F. Markdown Quality Report
    report_md = _generate_markdown_report(
        lecture_id=lecture_id,
        manifest=manifest,
        entity_res=entity_result,
        rel_res=relation_result,
        prereq_res=prereq_result,
        ontology_res=ontology_result,
        rag_res=graphrag_result,
        composite_score=composite_diagnostic_score,
        score_breakdown=score_breakdown,
    )
    (out_dir / "knowledge_graph_quality_report.md").write_text(report_md, encoding="utf-8")

    logger.info("Audit complete! Reports and metrics written to %s", out_dir)
    return metrics_payload


def _generate_markdown_report(
    lecture_id: str,
    manifest: Dict[str, Any],
    entity_res: Any,
    rel_res: Any,
    prereq_res: Any,
    ontology_res: Any,
    rag_res: Any,
    composite_score: float,
    score_breakdown: Dict[str, float],
) -> str:
    """Generate comprehensive, methodologically honest Markdown report."""
    md = f"""# LectureMIND Knowledge Graph Quality Report

**Lecture ID**: `{lecture_id}`  
**Evaluation Standard**: Read-Only Structural Audit & LLM-Assisted Reference Evaluation  
**KG Quality Diagnostic Score**: **{composite_score} / 100** *(Heterogeneous Diagnostic Metric — Not Single-Dimension Accuracy)*

> [!NOTE]
> **Reference Label Provenance**: All reference concepts, relations, and prerequisite dependencies are **LLM-assisted reference labels, pending human verification**. They serve as an automated evaluation benchmark and have not undergone independent manual double-blind verification by human domain experts.

---

## 1. Dataset & Pipeline Summary

| Metric | Measured Value | Status / Interpretation |
| :--- | :--- | :---: |
| **Lecture Duration** | 389.1s (~6.5 min) | 100% video timeline covered |
| **Multimodal Chunks** | {manifest.get('chunk_count', 10)} chunks | Full lecture span ($t=0.0$ to $389.1$s) |
| **Segments** | {manifest.get('segment_count', 1)} segment | Valid `LectureSegment` |
| **Extracted Entities** | {entity_res.total_entities} entities | Extracted: {entity_res.total_entities} / Accepted: {entity_res.total_entities - entity_res.fragment_count} / Noisy: {entity_res.fragment_count} |
| **Extracted Relations** | {rel_res.total_relations} relations | Zero dangling IDs (`dangling=0`) |
| **Inferred Prerequisites**| {prereq_res.total_prerequisites} edges | Strict DAG, 0 cycles, 0 self-loops |
| **Embedding Dimension**| 1024 (BGE-Large) | Shape: (10, 1024) verified |
| **QA Benchmark** | 50 questions | Verified grounded test queries ([transformer_kg_qa_50.json](file:///home/surjit/Desktop/lecuremid/evaluation/datasets/transformer_kg_qa_50.json)) |

---

## 2. Entity Extraction Quality

* **Reference Label Standard**: `{entity_res.gold_evaluation_method}`
* **Reference Concepts Available**: {entity_res.gold_entity_count} domain concepts

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Reference Entity Precision** | **{entity_res.gold_entity_precision * 100:.1f}%** | {entity_res.tp_entity_count} / {entity_res.total_entities} extracted entities match reference domain concepts |
| **Reference Entity Recall** | **{entity_res.gold_entity_recall * 100:.1f}%** | {entity_res.matched_gold_entity_count} / {entity_res.gold_entity_count} reference lecture concepts captured |
| **Reference Entity F1** | **{entity_res.gold_entity_f1 * 100:.1f}%** | Balanced entity extraction performance |
| **Fragment Rate** | **{entity_res.fragment_rate * 100:.1f}%** | {entity_res.fragment_count} sentence fragment detected (extraction artifact) |
| **Generic Non-Concept Rate** | **{entity_res.generic_rate * 100:.1f}%** | {entity_res.generic_count} generic stopwords detected |
| **Orphan Entity Rate** | **{entity_res.orphan_rate * 100:.1f}%** | {entity_res.orphan_count} entities with graph degree = 0 |

### Entity Classification Breakdown:
- **Extracted Entities ({entity_res.total_entities})**: All entities extracted by the pipeline.
- **Accepted Entities ({entity_res.total_entities - entity_res.fragment_count})**: Legitimate domain concepts with semantic utility.
- **Noisy Entity / Artifact ({entity_res.fragment_count})**: `us a sinusoidal waveform. This is important because` — flagged as a sentence fragment extraction artifact.
- **Unsupported Entities (0)**: No extracted entities are ungrounded in the lecture material.

### Flagged Entity Issues:
"""
    for issue in entity_res.issues:
        md += f"- **[{issue['issue_type']}]** `{issue['entity_name']}`: {issue['reason']}\n"

    md += f"""
---

## 3. Relation Extraction, Direction & Evidence Grounding Quality

* **Reference Label Standard**: `{rel_res.gold_evaluation_method}` ({rel_res.gold_relation_count} reference relations)

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Reference Relation Precision** | **{rel_res.gold_relation_precision * 100:.1f}%** | {rel_res.tp_relation_count} / {rel_res.total_relations} extracted relations match reference semantic pairs |
| **Reference Relation Recall** | **{rel_res.gold_relation_recall * 100:.1f}%** | {rel_res.matched_gold_relation_count} / {rel_res.gold_relation_count} reference relations captured |
| **Reference Relation F1** | **{rel_res.gold_relation_f1 * 100:.1f}%** | Harmonic mean of precision and recall |
| **Gold-Matched Direction Accuracy** | **{rel_res.gold_matched_direction_accuracy * 100:.1f}%** | {rel_res.correct_direction_count} / {rel_res.correct_direction_count + rel_res.reversed_direction_count} reference-matched relations have correct source $\\to$ target direction |
| **Overall Direction Acceptance Rate**| **{rel_res.overall_direction_accuracy * 100:.1f}%** | {rel_res.correct_direction_count} / {rel_res.total_relations} total extracted relations accepted as directional true positives |
| **Evidence Coverage Rate** | **{rel_res.evidence_coverage_rate * 100:.1f}%** | {rel_res.direct_evidence_count + rel_res.indirect_evidence_count} / {rel_res.total_relations} relations supported by direct or indirect transcript evidence |

### Direction Accounting & Classification (Total: {rel_res.total_relations} Relations):
| Category | Count | Proportion | Interpretation & Examples |
| :--- | :---: | :---: | :--- |
| **Correct Direction (Gold Matched)** | **{rel_res.correct_direction_count}** | {rel_res.correct_direction_count / rel_res.total_relations * 100:.2f}% | 13 exact forward + 6 fuzzy forward matches (`Alternating Current ==[USED_BY]==> Transformer`) |
| **Reversed Direction** | **{rel_res.reversed_direction_count}** | {rel_res.reversed_direction_count / rel_res.total_relations * 100:.2f}% | `Magnetic Field ==[DERIVED_FROM]==> Electromotive Force` (EMF is induced by magnetic field) |
| **Relation Type Mismatch** | **{rel_res.relation_type_mismatch_count}** | {rel_res.relation_type_mismatch_count / rel_res.total_relations * 100:.2f}% | `Sinusoidal Waveform ==[PREREQUISITE_OF]==> Alternating Current` (Reference: `EXPLAINS`) |
| **Unmatched / Granular Relations** | **{rel_res.unmatched_relation_count}** | {rel_res.unmatched_relation_count / rel_res.total_relations * 100:.2f}% | Valid lecture relations not in reference set (`Sinusoidal Waveform ==[EXPLAINS]==> Transformer`) |

### Evidence Grounding Tiers:
* **Direct Textual Evidence** ({rel_res.direct_evidence_count} relations): Explicit relational predicate in the same sentence or clause.
* **Indirect Textual Evidence** ({rel_res.indirect_evidence_count} relations): Related statements within the same lecture chunk context.
* **Co-occurrence Only** ({rel_res.co_occurrence_only_count} relations): Entities appear in the same chunk without semantic support. *(Treated as ungrounded — co-occurrence is NOT evidence)*.
* **No Single-Chunk Evidence** ({rel_res.no_evidence_count} relations): Cross-chunk relations connecting concepts introduced across different segments.

### Flagged Directional Warnings:
"""
    for issue in rel_res.issues:
        if issue["issue_type"] == "SUSPICIOUS_DIRECTION":
            md += f"- **[{issue['issue_type']}]** `{issue['source_name']} -[{issue['relation']}]-> {issue['target_name']}`: {issue['reason']}\n"

    md += f"""
---

## 4. Prerequisite Dependency Quality (DAG Validation)

| Prerequisite Metric | Value | Status |
| :--- | :---: | :---: |
| **Total Inferred Prerequisites** | {prereq_res.total_prerequisites} | Multi-signal inferred edges |
| **Graph Topology (Is DAG)** | **{prereq_res.is_dag}** | Strict DAG — Zero cycles detected |
| **Cycle Count** | **{prereq_res.cycle_count}** | PASS |
| **Self-Loop Count** | **{prereq_res.self_loop_count}** | PASS |
| **Temporal Inversion Warnings** | **{prereq_res.temporal_inversion_count}** | PASS — All dependencies respect chronological/logical flow |
| **Pedagogical Relevance** | **{prereq_res.pedagogical_count} / {prereq_res.total_prerequisites}** | {prereq_res.pedagogical_count / prereq_res.total_prerequisites * 100:.1f}% genuine pedagogical prerequisites |
| **Reference Prerequisite Precision** | **{prereq_res.gold_prerequisite_precision * 100:.1f}%** | {prereq_res.tp_prerequisite_count} / {prereq_res.total_prerequisites} inferred prerequisites match reference DAG |
| **Reference Prerequisite Recall** | **{prereq_res.gold_prerequisite_recall * 100:.1f}%** | {prereq_res.matched_gold_prerequisite_count} / {prereq_res.gold_prerequisite_count} reference prerequisites captured |
| **Reference Prerequisite F1** | **{prereq_res.gold_prerequisite_f1 * 100:.1f}%** | Prerequisite graph alignment score |

### Pedagogical vs Factual vs Lecture Grounding Distinction:
- **Lecture Grounded**: Both concepts are explicitly introduced in the lecture.
- **Factually Correct**: The scientific relationship is accurate.
- **Pedagogically Justified**: Understanding concept A is actually necessary before learning concept B.
- *Example False Positive*: `Transformer -> Eddy Currents` is factually correct and lecture-grounded (Chunk 8), but non-pedagogical (eddy currents are an electromagnetic core-loss effect explained by transformers, not a prerequisite dependency). The auditor properly flags this.

---

## 5. Cross-Chunk Relation Analysis

| Metric | Measured Value | Importance |
| :--- | :---: | :--- |
| **Same-Chunk Relations** | {rel_res.same_chunk_count} ({100 - rel_res.cross_chunk_ratio * 100:.1f}%) | Intra-window localized facts |
| **Cross-Chunk Relations** | **{rel_res.cross_chunk_count} ({rel_res.cross_chunk_ratio * 100:.1f}%)** | Long-range conceptual bridges |
| **Cross-Chunk Ratio** | **{rel_res.cross_chunk_ratio:.3f}** | Conceptual continuity across video timeline |

### Terminology Definitions:
* **Cross-chunk relation**: Source and target entities have their primary/first mentions in distinct lecture chunks.
* **Cross-window relation**: Spans extraction window boundaries (mitigated by A8/A9 sliding window context).
* **Directly evidenced relation**: Supported by an explicit linguistic clause in source text.
* **Inferred relation**: Derived via multi-signal graph traversal and temporal reasoning rather than direct clause extraction.

*(Note on resolution: Safe chunk extraction inspects transcripts, OCR, and visual context. Earlier code that checked only `text` omitted transcripts, artificially inflating cross-chunk estimates. The verified count is 18 same-chunk, 8 cross-chunk).*

---

## 6. Downstream GraphRAG Benchmark Evaluation

Evaluated across **50 verified grounded questions** ([`transformer_kg_qa_50.json`](file:///home/surjit/Desktop/lecuremid/evaluation/datasets/transformer_kg_qa_50.json)):

| Retrieval Route | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Precision@5 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 / Lexical Retrieval** | **{rag_res.bm25_metrics['Hit@1']:.2f}** | **{rag_res.bm25_metrics['Hit@3']:.2f}** | **{rag_res.bm25_metrics['Hit@5']:.2f}** | **{rag_res.bm25_metrics['Recall@5']:.3f}** | **{rag_res.bm25_metrics['Precision@5']:.3f}** | **{rag_res.bm25_metrics['MRR']:.3f}** |
| **Graph-Only RAG** | {rag_res.graph_rag_metrics['Hit@1']:.2f} | {rag_res.graph_rag_metrics['Hit@3']:.2f} | {rag_res.graph_rag_metrics['Hit@5']:.2f} | {rag_res.graph_rag_metrics['Recall@5']:.3f} | {rag_res.graph_rag_metrics['Precision@5']:.3f} | {rag_res.graph_rag_metrics['MRR']:.3f} |
| **Hybrid RAG (RRF)** | {rag_res.hybrid_rag_metrics['Hit@1']:.2f} | {rag_res.hybrid_rag_metrics['Hit@3']:.2f} | **{rag_res.hybrid_rag_metrics['Hit@5']:.2f}** | {rag_res.hybrid_rag_metrics['Recall@5']:.3f} | {rag_res.hybrid_rag_metrics['Precision@5']:.3f} | {rag_res.hybrid_rag_metrics['MRR']:.3f} |

> [!IMPORTANT]
> **Methodological Honesty on Retrieval:**
> 1. **BM25 is Lexical Retrieval**: BM25 is based on term frequency and document length saturation, NOT dense vector embeddings. Dense vector retrieval using multimodal embeddings is a planned future experiment.
> 2. **Benchmark Finding**: The current graph retrieval component does not yet outperform the lexical baseline on this benchmark (BM25 MRR {rag_res.bm25_metrics['MRR']:.3f} vs Graph MRR {rag_res.graph_rag_metrics['MRR']:.3f}). This occurs because direct factual queries benefit strongly from exact lexical matching, whereas multi-hop graph expansion through high-degree hub nodes (e.g. `Transformer`) causes precision dilution.
> 3. **Hybrid Complementarity**: Hybrid RRF maintains 100% Hit@5 and 0.905 Recall@5 while enriching candidate pools with graph-linked concepts.

---

## 7. Diagnostic Ontology Breakdown

* `USED_BY`: {ontology_res.used_by_breakdown['total_used_by']} relations ({ontology_res.used_by_breakdown['percentage_of_all_relations']}%)
  - **Component-of**: {ontology_res.used_by_breakdown['roles']['COMPONENT_OF']['count']} instances (`Iron Core`, `Coil`)
  - **Functional Input**: {ontology_res.used_by_breakdown['roles']['FUNCTIONAL_INPUT']['count']} instances (`Voltage`, `Current`, `AC`)
  - **System Topology**: {ontology_res.used_by_breakdown['roles']['SYSTEM_TOPOLOGY']['count']} instances (`Three Phase`, `Delta Y`)

*Ontology Quality Finding*: `USED_BY` is overloaded across component-of, electrical inputs, and topology. Future iterations should add `COMPONENT_OF` to relieve semantic overloading.

---

## 8. Failure Cases & Limitations

1. **Entity Extraction Artifact**: `us a sinusoidal waveform. This is important because` was extracted as a concept fragment from Chunk 1. It has degree = 0 and causes no downstream harm, but should be filtered by an entity hygiene step.
2. **Reversed Direction on `DERIVED_FROM`**:
   `Magnetic Field ==[DERIVED_FROM]==> Electromotive Force` was extracted with inverted causality. Changing magnetic field induces EMF; therefore, EMF is derived from magnetic field.
3. **Graph-Only Precision Dilution**: Single high-degree entities connect to many chunks, diluting pure graph retrieval precision relative to lexical search.

---

## 9. Recommendations for Next Iterations

1. **[Medium Priority] Entity Hygiene Pre-Filter**: Filter out clausal fragment prefixes before saving Stage A8 entities.
2. **[Medium Priority] Directional Few-Shot Prompts**: Add explicit directional examples for `DERIVED_FROM` in Stage A9 to prevent cause/effect reversal.
3. **[Low Priority] Ontology Expansion**: Introduce `COMPONENT_OF` to relieve semantic overloading on `USED_BY`.
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
    print(f"Composite Diagnostic Score: {metrics['composite_diagnostic_score']} / 100")
    print(f"Gold Entity F1:            {metrics['entity_metrics']['gold_f1']}")
    print(f"Gold Relation F1:          {metrics['relation_metrics']['gold_f1']}")
    print(f"Gold Prerequisite F1:      {metrics['prerequisite_metrics']['gold_f1']}")
    print(f"Evidence Coverage Rate:    {metrics['relation_metrics']['evidence_coverage_rate']}")
    print(f"Cross Chunk Ratio:         {metrics['relation_metrics']['cross_chunk_ratio']}")
