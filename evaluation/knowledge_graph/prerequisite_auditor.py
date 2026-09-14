# evaluation/knowledge_graph/prerequisite_auditor.py
# Read-only Prerequisite DAG Quality and Pedagogical Dependency Auditor.
#
# Audits prerequisite graphs produced by LectureMIND.
# Evaluates:
#   1. Strict DAG validity (Cycle detection via DFS)
#   2. Self-loop detection (A -> A)
#   3. Temporal causality analysis (t_A <= t_B as diagnostic warnings)
#   4. Pedagogical prerequisite vs Explanatory/Associative distinction
#   5. Unsupported prerequisite detection
#   6. Gold-standard comparison against prerequisite_gold.json

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PrerequisiteIssue:
    source_name: str
    target_name: str
    issue_type: str  # "CYCLE", "SELF_LOOP", "TEMPORAL_INVERSION", "UNSUPPORTED_PREREQUISITE", "NON_PEDAGOGICAL"
    reason: str
    severity: str    # "HIGH", "MEDIUM", "LOW"


@dataclass
class PrerequisiteAuditResult:
    total_prerequisites: int = 0
    is_dag: bool = True
    cycle_count: int = 0
    self_loop_count: int = 0
    temporal_inversion_count: int = 0
    unsupported_count: int = 0
    pedagogical_count: int = 0
    
    # Gold-standard metrics (None if gold data not evaluated)
    gold_prerequisite_precision: Optional[float] = None
    gold_prerequisite_recall: Optional[float] = None
    gold_prerequisite_f1: Optional[float] = None
    gold_evaluation_method: Optional[str] = None
    gold_prerequisite_count: int = 0
    matched_gold_prerequisite_count: int = 0
    tp_prerequisite_count: int = 0
    
    # Fuzzy/legacy metrics for transparent comparison with baseline
    fuzzy_prerequisite_precision: Optional[float] = None
    fuzzy_prerequisite_recall: Optional[float] = None
    fuzzy_prerequisite_f1: Optional[float] = None
    
    cycles_detected: List[List[str]] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    edge_details: List[Dict[str, Any]] = field(default_factory=list)


def _detect_cycles(edges: List[Tuple[str, str]]) -> List[List[str]]:
    """
    Find all simple cycles in a directed graph using DFS.
    Returns list of cycles, where each cycle is a list of node names [A, B, C, A].
    """
    adj: Dict[str, List[str]] = {}
    nodes: Set[str] = set()
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        nodes.add(u)
        nodes.add(v)

    visited: Dict[str, int] = {}  # 0: unvisited, 1: visiting (in stack), 2: finished
    for n in nodes:
        visited[n] = 0

    cycles: List[List[str]] = []
    path: List[str] = []

    def dfs(u: str):
        visited[u] = 1
        path.append(u)

        for v in adj.get(u, []):
            if visited[v] == 1:
                # Cycle found! Extract subpath from v to end of path
                idx = path.index(v)
                cycle = path[idx:] + [v]
                cycles.append(cycle)
            elif visited[v] == 0:
                dfs(v)

        path.pop()
        visited[u] = 2

    for n in nodes:
        if visited[n] == 0:
            dfs(n)

    return cycles


def audit_prerequisites(
    prerequisites: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    gold_prerequisites_path: Optional[str | Path] = None,
    non_pedagogical_target_markers: Optional[List[str]] = None,
) -> PrerequisiteAuditResult:
    """
    Perform a comprehensive read-only audit of prerequisite dependencies.

    Args:
        non_pedagogical_target_markers: Target-name substrings that mark an
            edge as explanatory/associative rather than a genuine pedagogical
            prerequisite (e.g. "eddy current" — a core-loss effect explained
            by transformers, not a foundational dependency). Domain-specific
            and optional: with an empty/None list no edge is flagged
            non-pedagogical by name alone.
    """
    total = len(prerequisites)
    entity_map = {e.get("entity_id", ""): e.get("name", "") for e in entities}
    
    # Map entity to earliest timestamp in lecture
    entity_earliest_t: Dict[str, float] = {}
    for e in entities:
        eid = e.get("entity_id", "")
        name = e.get("name", "").lower()
        t_min = 999999.0
        for c in chunks:
            txt = ((c.get("text") or "") + " " + (c.get("ocr_text") or "")).lower()
            if name in txt:
                t_min = min(t_min, c.get("start_time", 0.0))
        entity_earliest_t[eid] = t_min if t_min != 999999.0 else 0.0

    issues: List[PrerequisiteIssue] = []
    edge_details: List[Dict[str, Any]] = []

    if non_pedagogical_target_markers is None:
        non_pedagogical_target_markers = ["eddy current"]

    directed_edges: List[Tuple[str, str]] = []
    self_loop_count = 0
    temporal_inversion_count = 0
    unsupported_count = 0
    pedagogical_count = 0

    for p in prerequisites:
        src_id = p.get("source_id") or p.get("source_entity_id", "")
        tgt_id = p.get("target_id") or p.get("target_entity_id", "")
        src_name = p.get("source_name") or entity_map.get(src_id, src_id)
        tgt_name = p.get("target_name") or entity_map.get(tgt_id, tgt_id)

        # 1. Self-loop check
        if src_name == tgt_name or (src_id and src_id == tgt_id):
            self_loop_count += 1
            issues.append(PrerequisiteIssue(
                source_name=src_name,
                target_name=tgt_name,
                issue_type="SELF_LOOP",
                reason=f"Self-referential prerequisite edge '{src_name} -> {tgt_name}'.",
                severity="HIGH",
            ))
            continue

        directed_edges.append((src_name, tgt_name))

        # 2. Temporal order diagnostic check
        t_src = entity_earliest_t.get(src_id, 0.0)
        t_tgt = entity_earliest_t.get(tgt_id, 0.0)
        has_temporal_warning = False
        if t_src > t_tgt:
            has_temporal_warning = True
            temporal_inversion_count += 1
            issues.append(PrerequisiteIssue(
                source_name=src_name,
                target_name=tgt_name,
                issue_type="TEMPORAL_INVERSION",
                reason=(
                    f"Target '{tgt_name}' introduced earlier ({t_tgt:.1f}s) than source "
                    f"'{src_name}' ({t_src:.1f}s). Diagnostic warning: verify backward reference."
                ),
                severity="LOW",  # Diagnostic warning, not outright disqualifier per reviewer guidelines
            ))

        # 3. Evidence / Grounding support
        signals = p.get("signals", {})
        conf = p.get("confidence", 1.0)
        discourse = signals.get("discourse", 0.0)
        graph_sig = signals.get("graph", 0.0)

        is_supported = (conf >= 0.65) and (discourse > 0.0 or graph_sig > 0.0 or p.get("evidence_chunk_id") != "")
        if not is_supported and not p.get("is_inferred", False):
            unsupported_count += 1
            issues.append(PrerequisiteIssue(
                source_name=src_name,
                target_name=tgt_name,
                issue_type="UNSUPPORTED_PREREQUISITE",
                reason=f"Prerequisite edge has low confidence ({conf}) or lacks supporting discourse/graph signals.",
                severity="MEDIUM",
            ))

        # 4. Pedagogical vs Explanatory Distinction
        # E.g. Transformer -> Eddy Currents is explanatory (loss mechanism), not a strict conceptual necessity.
        # The marker list is caller-supplied so this check stays domain-neutral;
        # lectures without known non-pedagogical markers are never auto-flagged.
        is_explanatory_only = False
        if any(marker.lower() in tgt_name.lower() for marker in non_pedagogical_target_markers):
            is_explanatory_only = True
            issues.append(PrerequisiteIssue(
                source_name=src_name,
                target_name=tgt_name,
                issue_type="NON_PEDAGOGICAL",
                reason=(
                    f"'{src_name} -> {tgt_name}' represents an explanatory/associative relationship "
                    "(core loss phenomenon), rather than a foundational prerequisite dependency."
                ),
                severity="LOW",
            ))
        else:
            pedagogical_count += 1

        edge_details.append({
            "source_id": src_id,
            "source_name": src_name,
            "target_id": tgt_id,
            "target_name": tgt_name,
            "confidence": conf,
            "signals": signals,
            "t_source": t_src,
            "t_target": t_tgt,
            "has_temporal_warning": has_temporal_warning,
            "is_pedagogical": not is_explanatory_only,
            "is_supported": is_supported,
        })

    # 5. Cycle Detection
    cycles = _detect_cycles(directed_edges)
    is_dag = (len(cycles) == 0)
    for c in cycles:
        issues.append(PrerequisiteIssue(
            source_name=c[0],
            target_name=c[-1],
            issue_type="CYCLE",
            reason=f"Directed cycle detected: {' -> '.join(c)}",
            severity="HIGH",
        ))

    result = PrerequisiteAuditResult(
        total_prerequisites=total,
        is_dag=is_dag,
        cycle_count=len(cycles),
        self_loop_count=self_loop_count,
        temporal_inversion_count=temporal_inversion_count,
        unsupported_count=unsupported_count,
        pedagogical_count=pedagogical_count,
        cycles_detected=cycles,
        issues=[asdict(i) for i in issues],
        edge_details=edge_details,
    )

    # 6. Gold-Standard Prerequisite Evaluation
    if gold_prerequisites_path and Path(gold_prerequisites_path).exists():
        with open(gold_prerequisites_path, "r", encoding="utf-8") as fp:
            gold_data = json.load(fp)

        gold_list = gold_data.get("prerequisites", [])

        def _norm(name: str) -> str:
            return re.sub(r"\s+", " ", (name or "").lower().strip())

        gold_edges_norm = [
            (_norm(gp.get("source_name", "")), _norm(gp.get("target_name", "")), gp)
            for gp in gold_list
        ]

        # Only evaluate actual prerequisite edges (excluding any summary dicts)
        inferred_edges = [
            e for e in edge_details
            if isinstance(e, dict) and "source_name" in e and "target_name" in e
        ]

        # Sort inferred edges by confidence descending so highest confidence gets priority
        sorted_inferred = sorted(
            enumerate(inferred_edges),
            key=lambda item: item[1].get("confidence", 1.0),
            reverse=True,
        )

        # -------------------------------------------------------------
        # A. Strict Exact 1-to-1 Bipartite Matching
        # An inferred edge (s, t) matches gold (gs, gt) iff s_norm == gs_norm and t_norm == gt_norm.
        # Each gold edge can be matched at most once, and each inferred edge can match at most once.
        # -------------------------------------------------------------
        strict_matched_gold_indices = set()
        strict_tp_list = []
        strict_fp_list = []

        for inf_idx, edge in sorted_inferred:
            s_norm = _norm(edge.get("source_name", ""))
            t_norm = _norm(edge.get("target_name", ""))
            edge_str = f"{edge['source_name']} -> {edge['target_name']}"

            found_gold_idx = None
            for g_idx, (gs_norm, gt_norm, gp) in enumerate(gold_edges_norm):
                if g_idx in strict_matched_gold_indices:
                    continue
                if s_norm == gs_norm and t_norm == gt_norm:
                    found_gold_idx = g_idx
                    break

            if found_gold_idx is not None:
                strict_matched_gold_indices.add(found_gold_idx)
                strict_tp_list.append(edge_str)
            else:
                strict_fp_list.append(edge_str)

        strict_fn_list = [
            f"{gp.get('source_name')} -> {gp.get('target_name')}"
            for g_idx, (_, _, gp) in enumerate(gold_edges_norm)
            if g_idx not in strict_matched_gold_indices
        ]

        total_inf = len(inferred_edges)
        total_gold = len(gold_list)
        strict_tp = len(strict_tp_list)
        strict_prec = round(strict_tp / total_inf if total_inf > 0 else 0.0, 4)
        strict_rec = round(strict_tp / total_gold if total_gold > 0 else 0.0, 4)
        strict_f1 = round((2 * strict_prec * strict_rec / (strict_prec + strict_rec)) if (strict_prec + strict_rec) > 0 else 0.0, 4)

        # -------------------------------------------------------------
        # B. Fuzzy / Substring 1-to-1 Matching (for diagnostic comparison with legacy baseline)
        # Matches exact first, then allows partial substring overlap, 1-to-1 assignment.
        # -------------------------------------------------------------
        fuzzy_matched_gold_indices = set()
        fuzzy_tp_list = []
        fuzzy_fp_list = []

        for inf_idx, edge in sorted_inferred:
            s_norm = _norm(edge.get("source_name", ""))
            t_norm = _norm(edge.get("target_name", ""))
            edge_str = f"{edge['source_name']} -> {edge['target_name']}"

            found_gold_idx = None
            # Prioritize exact match
            for g_idx, (gs_norm, gt_norm, gp) in enumerate(gold_edges_norm):
                if g_idx in fuzzy_matched_gold_indices:
                    continue
                if s_norm == gs_norm and t_norm == gt_norm:
                    found_gold_idx = g_idx
                    break

            # Fallback to substring match
            if found_gold_idx is None:
                for g_idx, (gs_norm, gt_norm, gp) in enumerate(gold_edges_norm):
                    if g_idx in fuzzy_matched_gold_indices:
                        continue
                    if (s_norm == gs_norm or s_norm in gs_norm or gs_norm in s_norm) and \
                       (t_norm == gt_norm or t_norm in gt_norm or gt_norm in t_norm):
                        found_gold_idx = g_idx
                        break

            if found_gold_idx is not None:
                fuzzy_matched_gold_indices.add(found_gold_idx)
                fuzzy_tp_list.append(edge_str)
            else:
                fuzzy_fp_list.append(edge_str)

        fuzzy_fn_list = [
            f"{gp.get('source_name')} -> {gp.get('target_name')}"
            for g_idx, (_, _, gp) in enumerate(gold_edges_norm)
            if g_idx not in fuzzy_matched_gold_indices
        ]

        fuzzy_tp = len(fuzzy_tp_list)
        fuzzy_prec = round(fuzzy_tp / total_inf if total_inf > 0 else 0.0, 4)
        fuzzy_rec = round(fuzzy_tp / total_gold if total_gold > 0 else 0.0, 4)
        fuzzy_f1 = round((2 * fuzzy_prec * fuzzy_rec / (fuzzy_prec + fuzzy_rec)) if (fuzzy_prec + fuzzy_rec) > 0 else 0.0, 4)

        # Primary metrics report strict exact 1-to-1 matching
        result.gold_prerequisite_precision = strict_prec
        result.gold_prerequisite_recall = strict_rec
        result.gold_prerequisite_f1 = strict_f1
        result.gold_evaluation_method = (
            "Strict 1-to-1 exact matching against reference labels"
        )
        result.gold_prerequisite_count = total_gold
        result.matched_gold_prerequisite_count = strict_tp
        result.tp_prerequisite_count = strict_tp

        result.fuzzy_prerequisite_precision = fuzzy_prec
        result.fuzzy_prerequisite_recall = fuzzy_rec
        result.fuzzy_prerequisite_f1 = fuzzy_f1

        result.edge_details.append({
            "tp_prerequisites": strict_tp_list,
            "fp_prerequisites": strict_fp_list,
            "fn_prerequisites": strict_fn_list,
            "strict_metrics": {
                "tp": strict_tp,
                "fp": len(strict_fp_list),
                "fn": len(strict_fn_list),
                "precision": strict_prec,
                "recall": strict_rec,
                "f1": strict_f1,
            },
            "fuzzy_metrics": {
                "tp": fuzzy_tp,
                "fp": len(fuzzy_fp_list),
                "fn": len(fuzzy_fn_list),
                "precision": fuzzy_prec,
                "recall": fuzzy_rec,
                "f1": fuzzy_f1,
                "tp_prerequisites": fuzzy_tp_list,
                "fp_prerequisites": fuzzy_fp_list,
                "fn_prerequisites": fuzzy_fn_list,
            },
        })

    return result
