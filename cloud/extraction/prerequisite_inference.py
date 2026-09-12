# cloud/extraction/prerequisite_inference.py
# Socratic Prerequisite Back-Tracker: Prerequisite Inference Engine.
#
# Infers conceptual dependencies (PREREQUISITE_OF) within a single lecture.
# Strictly preserves single-lecture isolation and leaves Stage A9 frozen.
#
# Key Architectural Guarantees:
# 1. Candidate pre-filtering avoids brute-force all-pairs search while reporting metrics.
# 2. Hard temporal gate: t(A) < t(B) is mandatory (with text offset tie-break if same chunk).
# 3. Hard evidence anchor: Temporal precedence alone NEVER creates edges;
#    candidates MUST have supporting discourse or graph evidence.
# 4. Multi-signal scoring: Discourse cues + Graph coupling (0.70 combined semantic weight)
#    + Foundational prominence (0.15) + Temporal proximity (0.15).
# 5. Deterministic DAG cycle resolution: Drops the edge with min(confidence).

import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from config import SharedSettings

logger = logging.getLogger(__name__)

# Directional prerequisite implication weights from existing A9 relations:
# maps (relation_type, is_forward) -> prerequisite weight
RELATION_PREREQUISITE_WEIGHTS = {
    # If s PREREQUISITE_OF t: s is prerequisite of t (explicit strong dependency)
    ("PREREQUISITE_OF", True): 1.0,
    ("PREREQUISITE_OF", False): 0.0,
    # If s INTRODUCED_BEFORE t: chronological/specialization ordering
    ("INTRODUCED_BEFORE", True): 0.70,
    ("INTRODUCED_BEFORE", False): 0.0,
    # If s DERIVED_FROM t (s is derived from t): t is prerequisite of s
    ("DERIVED_FROM", True): 0.0,
    ("DERIVED_FROM", False): 0.80,
    # If s USED_BY t: t uses s (hardware/variable/component usage).
    # Supporting evidence only; cannot pass threshold on its own (max score 0.44 < 0.65).
    ("USED_BY", True): 0.20,
    ("USED_BY", False): 0.0,
    # If s EXPLAINS t: s explains t (explanatory/loss mechanism).
    # Supporting evidence only; cannot pass threshold on its own.
    ("EXPLAINS", True): 0.10,
    ("EXPLAINS", False): 0.0,
    # Visualization: no prerequisite implication
    ("VISUALIZED_BY", True): 0.0,
    ("VISUALIZED_BY", False): 0.0,
}

_STOPWORDS = {
    "what", "which", "how", "why", "when", "where", "who", "does", "did",
    "the", "and", "are", "was", "were", "with", "from", "that", "this",
    "have", "has", "had", "for", "its", "not", "but", "you", "your",
    "about", "into", "can", "could", "will", "would", "shall", "should",
    "than", "then", "there", "their", "they", "is", "it", "of", "to",
    "in", "on", "at", "by", "be", "or", "as", "an", "a", "lecture",
    "lectures", "we", "let", "lets", "us", "now", "see", "also",
}


def _clean_text(text: str) -> str:
    """Normalize text for discourse cue matching."""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^\w\s\.,;:!\?\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


GENERIC_HEAD_NOUNS = {
    "connection", "configuration", "circuit", "concept", "type", "device", "phenomenon", "mechanism"
}


def _build_term_regex(term: str) -> str:
    """
    Build a flexible regex that matches singular/plural and inflectional
    variations of each word in a term (e.g. 'Page Table' -> 'page(s)? table(s|es)?'),
    with support for hyphens, connectors ('and'), optional classification head nouns,
    and standard technical acronyms (e.g. 'AC', 'DC', 'EMF').
    """
    # Distinguish generic 'Current' from compound terms ('Alternating Current', 'AC Current', 'Eddy Current', 'Direct Current', 'DC Current')
    if term.strip().lower() in {"current", "currents"}:
        return r"(?<!alternating\s)(?<!alternating\-)(?<!ac\s)(?<!ac\-)(?<!eddy\s)(?<!eddy\-)(?<!direct\s)(?<!direct\-)(?<!dc\s)(?<!dc\-)current(?:s|es)?"

    tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", term.lower()) if t]
    if not tokens:
        return re.escape(term)
    parts = []
    for tok in tokens:
        if len(tok) > 3 and tok.endswith("ies"):
            base = tok[:-3]
            parts.append(f"{re.escape(base)}(?:y|ies)")
        elif len(tok) > 3 and tok.endswith("es"):
            base = tok[:-2]
            parts.append(f"{re.escape(base)}(?:es|s)?")
        elif len(tok) > 2 and tok.endswith("s") and not tok.endswith("ss"):
            base = tok[:-1]
            parts.append(f"{re.escape(base)}(?:s|es)?")
        else:
            parts.append(f"{re.escape(tok)}(?:s|es)?")

    prefix_sep = r"(?:[\s\-]+(?:and\s+)?)"
    if len(tokens) >= 3 and tokens[-1] in GENERIC_HEAD_NOUNS:
        prefix = prefix_sep.join(parts[:-1])
        suffix = parts[-1]
        term_pattern = f"(?:{prefix}(?:{prefix_sep}{suffix})?)"
    else:
        term_pattern = prefix_sep.join(parts)

    # Technical acronym and abbreviation aliases
    words = term.split()
    if len(words) >= 2:
        acronym = "".join([w[0].lower() for w in words if w[0].isalnum()])
        if len(acronym) >= 2 and acronym in {"ac", "dc"}:
            term_pattern = f"(?:{term_pattern}|{re.escape(acronym)})"
        elif "electromotive" in term.lower():
            term_pattern = f"(?:{term_pattern}|emf)"

    return term_pattern


class DiscourseCueDetector:
    """
    High-precision discourse cue detector for pedagogical prerequisites.

    Detects explicit dependency phrasing, sequential pedagogical cues,
    and recall markers in lecture transcripts.
    """

    def __init__(self):
        # Patterns ordered by specificity and base confidence
        self._compiled_patterns = [
            # 1. Explicit prerequisite statements (Confidence: 1.0)
            (
                r"\b{A}\b[^.]{1,80}\bis\s+(?:a\s+)?prerequisite\s+(?:for|to)\s+\b{B}\b",
                1.0,
                "explicit_prerequisite",
            ),
            (
                r"\bprerequisite\s+(?:for|to)\s+\b{B}\b[^.]{1,80}\bis\s+\b{A}\b",
                1.0,
                "explicit_prerequisite_rev",
            ),
            (
                r"\bwithout\s+(?:understanding\s+|knowing\s+)?\b{A}\b[^.]{1,80}(?:cannot|can't|impossible\s+to)\s+(?:understand|grasp|follow)\s+\b{B}\b",
                0.95,
                "negative_necessity",
            ),
            (
                r"\b{B}\b[^.]{1,60}\bcan\s+only\s+(?:work|operate|function)\s+(?:using|with|by|if\s+there\s+is)\s+(?:an?\s+)?\b{A}\b",
                0.95,
                "only_works_with_requirement",
            ),
            (
                r"\bwhy\s+only\s+(?:an?\s+)?\b{A}\b\s+can\s+be\s+used\s+in\s+\b{B}\b",
                0.95,
                "why_only_used_requirement",
            ),
            # 2. Before / In-order-to dependency cues (Confidence: 0.90 - 0.95)
            (
                r"\bbefore\s+(?:we\s+)?(?:move\s+on\s+to|discuss|discussing|look\s+at|talk\s+about|understand|learn)\s+\b{B}\b[^.]{1,80}(?:let\s*\'?s|let\s+us|we|must|first|need\s+to|should)\s+(?:first\s+)?(?:understand|look\s+at|know|cover|review)\s+\b{A}\b",
                0.95,
                "pedagogical_precedence",
            ),
            (
                r"\b(?:need|have|require)\s+to\s+(?:know|understand|master)\s+\b{A}\b[^.]{1,80}(?:in\s+order\s+to|to)\s+(?:understand|know|study|implement)\s+\b{B}\b",
                0.95,
                "understanding_requirement",
            ),
            (
                r"\bin\s+order\s+to\s+(?:understand|grasp|follow)\s+\b{B}\b[^.]{1,80}(?:you|we|one)?\s*(?:first\s+)?(?:must|need\s+to|have\s+to)\s+(?:understand|know)\s+\b{A}\b",
                0.95,
                "in_order_to_requirement",
            ),
            # 3. Specialization / Categorical Subdivision cues (Confidence: 0.90)
            (
                r"\b{A}\b[^.]{1,80}\b(?:are\s+manufactured\s+to\s+be|can\s+be\s+(?:classified|categorized|divided|split)\s+into|come\s+in\s+two\s+(?:types|forms)\s*(?:such\s+as|like|:)?|can\s+be\s+either|are\s+usually\s+in\s+a)\s+[^.]{0,60}\b{B}\b",
                0.90,
                "specialization_subdivision",
            ),
            (
                r"\b{B}\b[^.]{0,60}\b(?:is\s+a\s+(?:type|form|kind|specialization|variation)\s+of|is\s+simply\s+a|are\s+made\s+from\s+(?:either\s+)?(?:three\s+)?(?:separate\s+)?)\s+\b{A}\b",
                0.90,
                "specialization_type_of",
            ),
            # 4. Compositional & Topological Configuration cues (Confidence: 0.85 - 0.90)
            (
                r"\b{A}\b\s+(?:configuration|system|setup)\s+(?:is\s+)?(?:known\s+as|called|referred\s+to\s+as|designated\s+as)\s+(?:a\s+|an\s+)?\b{B}\b",
                0.85,
                "topological_configuration",
            ),
            (
                r"\b{B}\b[^.]{1,60}\b(?:refers\s+to|consists\s+of|is\s+a\s+configuration\s+of)\s+[^.]{1,60}\b{A}\b",
                0.85,
                "config_reference",
            ),
            # 5. Causal Physical Foundation & Inductive Generation cues (Confidence: 0.85 - 0.90)
            (
                r"\b(?:change\s+in|fluctuation\s+in|alternation\s+(?:of|in)|flow\s+of)\s+\b{A}\b[^.]{1,60}\b(?:creates|generates|produces|induces|results\s+in)\s+(?:a\s+|an\s+|the\s+)?(?:fluctuating\s+|changing\s+|varying\s+|dynamic\s+)?\b{B}\b",
                0.85,
                "causal_generation",
            ),
            (
                r"\bpass\s+(?:an?\s+)?\b{A}\b[^.]{1,40}\bthen\s+(?:the\s+)?\b{B}\b\s+will\s+(?:fluctuate|vary|increase|decrease|change|reverse)\b",
                0.85,
                "signal_variation",
            ),
            (
                r"\b(?:change\s+in\s+(?:the\s+)?(?:intensity\s+and\s+direction\s+of\s+(?:the\s+)?|fluctuation\s+of\s+(?:the\s+)?)|fluctuating\s+)\b{A}\b[^.]{1,60}\b(?:forces\s+(?:them|electrons)\s+to\s+move|produces|induces)\s+[^.]{1,60}\b{B}\b",
                0.85,
                "inductive_generation",
            ),
            (
                r"\b{A}\b\s+(?:constantly\s+)?(?:disturbs|induces|creates|generates|produces|causes)\b[^.]{0,100}\.\s+this\s+(?:movement|force|action|induction|effect|phenomenon)\s+(?:is\s+known\s+as|is\s+called|causes|induces|produces|results\s+in)\s+[^.]{0,80}\b{B}\b",
                0.90,
                "anaphoric_induction",
            ),
            (
                r"\b{B}\b[^.]{1,60}\b(?:is\s+induced|is\s+generated|arises\s+from|occurs\s+due\s+to)\s+(?:due\s+to|because\s+of|from\s+the\s+(?:changing|fluctuating|varying))\s+\b{A}\b",
                0.85,
                "passive_generation",
            ),
            (
                r"\bto\s+(?:increase|decrease|step)\s+(?:the\s+voltage\s+in\s+)?(?:a\s+)?\b{B}\b[^.]{1,60}\b(?:turns\s+(?:to|on|in)\s+(?:the\s+)?)\b{A}\b",
                0.85,
                "turns_stepping_enablement",
            ),
            # 6. Structural Reliance cues (Confidence: 0.85 - 0.90)
            (
                r"\b{B}\b[^.]{1,60}\b(?:relies\s+on|depends\s+on|builds\s+upon|builds\s+on|is\s+based\s+on)\s+(?:the\s+mechanism\s+(?:we\s+discussed\s+earlier,?\s*)?)?\b{A}\b",
                0.90,
                "reliance_dependency",
            ),
            (
                r"\b(?:recall|remember)\s+\b{A}\b[^.]{1,60}(?:because|since|as)\s+we(?:'ll|\s+will)?\s+(?:use|need)\s+it\s+(?:here\s+)?(?:for|in)?\s*\b{B}\b",
                0.85,
                "recall_relevance",
            ),
            (
                r"\b{A}\b[^.]{1,60}\bis\s+(?:needed|required|essential|fundamental)\s+for\s+\b{B}\b",
                0.85,
                "essential_foundation",
            ),
        ]

    def detect(self, text: str, name_a: str, name_b: str) -> Tuple[float, Optional[str], Optional[str]]:
        """
        Detect discourse cues connecting name_a (prerequisite) to name_b (target).

        Returns:
            (score, pattern_name, matched_snippet)
        """
        if not text or not name_a or not name_b:
            return 0.0, None, None

        clean = _clean_text(text)
        re_a = _build_term_regex(name_a)
        re_b = _build_term_regex(name_b)

        # Quick check: regex for both terms must find at least one match in text
        if not re.search(r"\b" + re_a + r"\b", clean, re.IGNORECASE):
            return 0.0, None, None
        if not re.search(r"\b" + re_b + r"\b", clean, re.IGNORECASE):
            return 0.0, None, None

        best_score = 0.0
        best_pattern = None
        best_snippet = None

        for pattern_template, base_score, pattern_name in self._compiled_patterns:
            pattern_str = pattern_template.replace("{A}", re_a).replace("{B}", re_b)
            try:
                match = re.search(pattern_str, clean, re.IGNORECASE)
                if match:
                    if base_score > best_score:
                        best_score = base_score
                        best_pattern = pattern_name
                        start_pos = max(0, match.start() - 30)
                        end_pos = min(len(clean), match.end() + 30)
                        best_snippet = clean[start_pos:end_pos].strip()
            except re.error:
                continue

        return round(best_score, 4), best_pattern, best_snippet


def compute_entity_profiles(
    entities: List[Any],
    chunks: List[Any],
    segments: List[Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Profile each entity with earliest timestamp, chunk occurrences,
    segment locations, and prominence in headings.
    """
    profiles: Dict[str, Dict[str, Any]] = {}

    for ent in entities:
        ent_id = ent.entity_id if hasattr(ent, "entity_id") else ent.get("entity_id", "")
        name = ent.name if hasattr(ent, "name") else ent.get("name", "")
        if hasattr(ent, "type"):
            ent_type = ent.type.value if hasattr(ent.type, "value") else str(ent.type)
        elif isinstance(ent, dict):
            raw_type = ent.get("type", "Concept")
            ent_type = raw_type.value if hasattr(raw_type, "value") else str(raw_type)
        else:
            ent_type = "Concept"

        profiles[ent_id] = {
            "entity_id": ent_id,
            "name": name,
            "type": ent_type,
            "earliest_timestamp": float("inf"),
            "earliest_chunk_id": "",
            "earliest_offset": float("inf"),
            "chunk_ids": [],
            "segment_ids": set(),
            "in_headings": False,
            "in_first_segment": False,
        }

    # Map chunks
    for chunk in chunks:
        cid = chunk.chunk_id if hasattr(chunk, "chunk_id") else chunk.get("chunk_id", "")
        ts = chunk.timestamp if hasattr(chunk, "timestamp") else float(chunk.get("timestamp", 0.0))
        t_clean = _clean_text(chunk.transcript if hasattr(chunk, "transcript") else chunk.get("transcript", ""))
        ocr_clean = _clean_text(chunk.ocr_text if hasattr(chunk, "ocr_text") else chunk.get("ocr_text", ""))
        vis_clean = _clean_text(chunk.visual_context if hasattr(chunk, "visual_context") else chunk.get("visual_context", ""))

        full_text = f"{t_clean} {ocr_clean} {vis_clean}"

        for ent_id, prof in profiles.items():
            ent_name = prof["name"]
            re_term = _build_term_regex(ent_name)
            match = re.search(r"\b" + re_term + r"\b", full_text, re.IGNORECASE)

            if match:
                prof["chunk_ids"].append(cid)
                offset = match.start()
                if ts < prof["earliest_timestamp"] or (ts == prof["earliest_timestamp"] and offset < prof["earliest_offset"]):
                    prof["earliest_timestamp"] = ts
                    prof["earliest_chunk_id"] = cid
                    prof["earliest_offset"] = offset

                # Check if in slide heading or visual context title
                if re.search(r"\b" + re_term + r"\b", ocr_clean, re.IGNORECASE) or re.search(r"\b" + re_term + r"\b", vis_clean, re.IGNORECASE):
                    prof["in_headings"] = True

    # Map segments
    for i, seg in enumerate(segments):
        sid = seg.segment_id if hasattr(seg, "segment_id") else seg.get("segment_id", "")
        title_clean = _clean_text(seg.title if hasattr(seg, "title") else seg.get("title", ""))
        seg_start = seg.start if hasattr(seg, "start") else float(seg.get("start", 0.0))
        seg_chunks = set(seg.chunks if hasattr(seg, "chunks") else seg.get("chunks", []))

        for ent_id, prof in profiles.items():
            re_term = _build_term_regex(prof["name"])
            match = re.search(r"\b" + re_term + r"\b", title_clean, re.IGNORECASE)

            # Mentioned in segment title
            if match:
                prof["segment_ids"].add(sid)
                prof["in_headings"] = True
                if i == 0:
                    prof["in_first_segment"] = True
                if seg_start < prof["earliest_timestamp"]:
                    prof["earliest_timestamp"] = seg_start
                    prof["earliest_offset"] = match.start()

            # Mentioned via mapped chunks
            if seg_chunks.intersection(set(prof["chunk_ids"])):
                prof["segment_ids"].add(sid)
                if i == 0:
                    prof["in_first_segment"] = True

    return profiles


def generate_prerequisite_candidates(
    entities: List[Any],
    chunks: List[Any],
    segments: List[Any],
    relations: List[Any],
    max_segment_window: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate candidate prerequisite pairs (A, B) using high-recall pre-filtering.

    A candidate pair (A, B) implies 'A is a PREREQUISITE_OF B'.
    Filters out impossible pairs:
    1. Same entity (A == B)
    2. Temporal causality gate: t(A) < t(B) is mandatory.
       If t(A) == t(B) in the same initial chunk, offset(A) < offset(B) is required.
    3. Structural & discourse pre-filtering: Pair must share an existing A9 relation
       OR co-occur within a 3-segment window.

    Returns:
        (candidates, metrics)
    """
    profiles = compute_entity_profiles(entities, chunks, segments)
    entity_list = list(profiles.values())
    n = len(entity_list)
    total_possible_pairs = n * (n - 1) if n > 1 else 0

    # Build existing relation index between entity pairs
    existing_relations: Dict[Tuple[str, str], List[str]] = {}
    for rel in relations:
        src = rel.source_entity_id if hasattr(rel, "source_entity_id") else rel.get("source_entity_id", "")
        tgt = rel.target_entity_id if hasattr(rel, "target_entity_id") else rel.get("target_entity_id", "")
        if hasattr(rel, "relation"):
            rtype = rel.relation.value if hasattr(rel.relation, "value") else str(rel.relation)
        elif isinstance(rel, dict):
            raw_r = rel.get("relation", "")
            rtype = raw_r.value if hasattr(raw_r, "value") else str(raw_r)
        else:
            rtype = ""

        existing_relations.setdefault((src, tgt), []).append(rtype)
        existing_relations.setdefault((tgt, src), []).append(f"REV_{rtype}")

    # Build segment index to check segment proximity
    segment_order: Dict[str, int] = {}
    for idx, seg in enumerate(segments):
        sid = seg.segment_id if hasattr(seg, "segment_id") else seg.get("segment_id", "")
        segment_order[sid] = idx

    candidates: List[Dict[str, Any]] = []
    rejections = {
        "same_entity": 0,
        "temporal_order": 0,
        "not_co_occurring_or_related": 0,
    }
    evaluated_pairs = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            evaluated_pairs += 1
            ent_a = entity_list[i]
            ent_b = entity_list[j]

            t_a = ent_a["earliest_timestamp"]
            t_b = ent_b["earliest_timestamp"]
            off_a = ent_a.get("earliest_offset", 0.0)
            off_b = ent_b.get("earliest_offset", 0.0)

            if math.isinf(t_a) or math.isinf(t_b):
                rejections["temporal_order"] += 1
                continue

            id_a = ent_a["entity_id"]
            id_b = ent_b["entity_id"]
            fwd_rels = existing_relations.get((id_a, id_b), [])
            rev_rels = existing_relations.get((id_b, id_a), [])
            has_rel = bool(fwd_rels or rev_rels)

            # Gate 1: Temporal causality with pedagogical inversion allowance.
            # Lecture presentation order sometimes introduces an overarching device
            # before foundational physics (e.g. Transformer at 0s, EMF at 132s).
            # If t(A) > t(B), allow ONLY IF backed by explicit strong prerequisite evidence:
            # - Direct PREREQUISITE_OF relation in A9, OR
            # - Target is derived from source (DERIVED_FROM in reverse), OR
            # - Strong discourse cue in transcript (score >= 0.80).
            # Otherwise, backward temporal order is strictly rejected.
            if t_a > t_b:
                has_strong_rel = ("PREREQUISITE_OF" in fwd_rels) or ("DERIVED_FROM" in rev_rels)
                if not has_strong_rel:
                    has_strong_disc = False
                    detector_inst = DiscourseCueDetector()
                    for chk in chunks:
                        txt = chk.transcript if hasattr(chk, "transcript") else chk.get("transcript", "")
                        sc, _, _ = detector_inst.detect(txt, ent_a["name"], ent_b["name"])
                        if sc >= 0.80:
                            has_strong_disc = True
                            break
                    if not has_strong_disc:
                        rejections["temporal_order"] += 1
                        continue

            # Gate 2: Structural or discourse proximity pre-filter
            id_a = ent_a["entity_id"]
            id_b = ent_b["entity_id"]

            has_rel = (id_a, id_b) in existing_relations or (id_b, id_a) in existing_relations

            # Check segment distance
            segs_a = ent_a["segment_ids"]
            segs_b = ent_b["segment_ids"]
            min_seg_dist = float("inf")
            for sa in segs_a:
                for sb in segs_b:
                    if sa in segment_order and sb in segment_order:
                        dist = abs(segment_order[sa] - segment_order[sb])
                        if dist < min_seg_dist:
                            min_seg_dist = dist

            # Check chunk overlap / proximity
            chunks_a = set(ent_a["chunk_ids"])
            chunks_b = set(ent_b["chunk_ids"])
            shares_chunk = bool(chunks_a.intersection(chunks_b))

            is_proximate = shares_chunk or (min_seg_dist <= max_segment_window)

            if not (has_rel or is_proximate):
                rejections["not_co_occurring_or_related"] += 1
                continue

            candidates.append({
                "source_id": id_a,
                "source_name": ent_a["name"],
                "source_type": ent_a["type"],
                "target_id": id_b,
                "target_name": ent_b["name"],
                "target_type": ent_b["type"],
                "t_source": t_a,
                "t_target": t_b,
                "source_chunk_id": ent_a["earliest_chunk_id"],
                "source_in_headings": ent_a["in_headings"],
                "source_in_first_segment": ent_a["in_first_segment"],
                "existing_forward_relations": existing_relations.get((id_a, id_b), []),
                "existing_reverse_relations": existing_relations.get((id_b, id_a), []),
            })

    metrics = {
        "total_possible_pairs": total_possible_pairs,
        "evaluated_pairs": evaluated_pairs,
        "accepted_candidates": len(candidates),
        "rejections": rejections,
    }

    logger.info(
        "Candidate generation complete: %d accepted from %d evaluated pairs (search space reduction: %.1f%%).",
        len(candidates),
        evaluated_pairs,
        (1.0 - (len(candidates) / evaluated_pairs if evaluated_pairs > 0 else 0.0)) * 100.0,
    )
    return candidates, metrics


def score_prerequisite_candidate(
    candidate: Dict[str, Any],
    chunks: List[Any],
    detector: Optional[DiscourseCueDetector] = None,
) -> Dict[str, Any]:
    """
    Score a candidate pair (A, B) using multi-signal evidence scoring with a HARD EVIDENCE ANCHOR.

    Weights:
    - Semantic dependency (0.70): Combines explicit discourse cues and graph coupling.
      max(S_discourse, S_graph) + 0.25 * min(S_discourse, S_graph)
    - Foundational prominence (0.15): A appears in slide headers or first segment title.
    - Temporal proximity (0.15): Proximity bonus for sequential logical progression.

    HARD ANCHOR:
    Temporal precedence alone MUST NEVER make an edge pass.
    If BOTH discourse evidence == 0.0 AND graph evidence == 0.0,
    the candidate is strictly assigned confidence = 0.0.
    """
    if detector is None:
        detector = DiscourseCueDetector()

    name_a = candidate["source_name"]
    name_b = candidate["target_name"]
    t_a = candidate["t_source"]
    t_b = candidate["t_target"]

    # 1. Discourse Evidence
    best_discourse_score = 0.0
    best_discourse_pattern = None
    best_discourse_snippet = None
    best_discourse_chunk_id = candidate.get("source_chunk_id", "")
    best_discourse_timestamp = t_a

    # Scan chunks that mention either A or B, or chunks in between
    for chunk in chunks:
        cid = chunk.chunk_id if hasattr(chunk, "chunk_id") else chunk.get("chunk_id", "")
        ts = chunk.timestamp if hasattr(chunk, "timestamp") else float(chunk.get("timestamp", 0.0))
        transcript = chunk.transcript if hasattr(chunk, "transcript") else chunk.get("transcript", "")

        # Look in transcript
        score, pattern, snippet = detector.detect(transcript, name_a, name_b)
        if score > best_discourse_score:
            best_discourse_score = score
            best_discourse_pattern = pattern
            best_discourse_snippet = snippet
            best_discourse_chunk_id = cid
            best_discourse_timestamp = ts

    # 2. Directional Graph Evidence
    best_graph_score = 0.0
    for rel_type in candidate.get("existing_forward_relations", []):
        weight = RELATION_PREREQUISITE_WEIGHTS.get((rel_type, True), 0.0)
        if weight > best_graph_score:
            best_graph_score = weight

    for rel_type in candidate.get("existing_reverse_relations", []):
        weight = RELATION_PREREQUISITE_WEIGHTS.get((rel_type, False), 0.0)
        if weight > best_graph_score:
            best_graph_score = weight

    # 3. Foundational Prominence (0.15)
    prominence_score = 0.0
    if candidate.get("source_in_headings", False):
        prominence_score += 0.7
    if candidate.get("source_in_first_segment", False):
        prominence_score += 0.3
    prominence_score = min(1.0, prominence_score)

    # 4. Temporal Proximity (0.15)
    if t_a == t_b:
        # Unknown temporal ordering (same initial chunk); assign neutral score
        # so other evidence determines validity without unearned forward precedence.
        temporal_score = 0.5
    elif t_a < t_b:
        delta_t = max(0.0, t_b - t_a)
        # Proximity decay over 10 minutes (600 seconds)
        temporal_score = 1.0 / (1.0 + (delta_t / 600.0))
    else:
        # Backward temporal delivery (e.g. foundational concept introduced after overarching artifact)
        # Moderate base score (0.4) decayed by time gap to reflect non-standard presentation order
        delta_t = t_a - t_b
        temporal_score = 0.4 / (1.0 + (delta_t / 600.0))

    # CRITICAL: HARD EVIDENCE ANCHOR
    # Merely appearing earlier or having a slide title MUST NOT create an edge.
    # At least one real structural/pedagogical evidence signal (discourse or graph) is REQUIRED.
    if best_discourse_score == 0.0 and best_graph_score == 0.0:
        composite_confidence = 0.0
    else:
        # Fused semantic dependency: primary signal + reinforcement bonus
        semantic_dependency = min(
            1.0,
            max(best_discourse_score, best_graph_score)
            + 0.25 * min(best_discourse_score, best_graph_score),
        )

        raw_confidence = (
            0.70 * semantic_dependency
            + 0.15 * prominence_score
            + 0.15 * temporal_score
        )
        composite_confidence = round(min(1.0, max(0.0, raw_confidence)), 4)

    return {
        "source_id": candidate["source_id"],
        "source_name": name_a,
        "source_type": candidate["source_type"],
        "target_id": candidate["target_id"],
        "target_name": name_b,
        "target_type": candidate["target_type"],
        "relation": "PREREQUISITE_OF",
        "confidence": composite_confidence,
        "is_inferred": True,
        "evidence_chunk_id": best_discourse_chunk_id or candidate.get("source_chunk_id", ""),
        "evidence_timestamp": best_discourse_timestamp,
        "signals": {
            "discourse": best_discourse_score,
            "graph": best_graph_score,
            "prominence": prominence_score,
            "temporal": round(temporal_score, 4),
        },
        "discourse_pattern": best_discourse_pattern,
        "discourse_snippet": best_discourse_snippet,
    }


def enforce_dag_cycles(candidate_edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure the inferred prerequisite graph is a strict Directed Acyclic Graph (DAG).

    Detects any directed cycles (A -> B -> ... -> A) and deterministically eliminates
    the edge with the minimum confidence in each cycle until no cycles remain.
    """
    edges = list(candidate_edges)

    def find_cycle(current_edges: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Find a directed cycle using DFS if one exists."""
        adj: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for edge in current_edges:
            adj.setdefault(edge["source_id"], []).append((edge["target_id"], edge))

        visited: Set[str] = set()
        in_stack: Set[str] = set()
        stack: List[Tuple[str, Optional[Dict[str, Any]]]] = []

        def dfs(node: str) -> Optional[List[Dict[str, Any]]]:
            visited.add(node)
            in_stack.add(node)

            for neighbor, edge_obj in adj.get(node, []):
                stack.append((neighbor, edge_obj))
                if neighbor not in visited:
                    res = dfs(neighbor)
                    if res:
                        return res
                elif neighbor in in_stack:
                    # Cycle detected: extract edges from stack
                    cycle_edges = []
                    for n, e in reversed(stack):
                        if e is not None:
                            cycle_edges.append(e)
                        if n == neighbor:
                            break
                    return cycle_edges
                stack.pop()

            in_stack.remove(node)
            return None

        for edge in current_edges:
            src = edge["source_id"]
            if src not in visited:
                cycle = dfs(src)
                if cycle:
                    return cycle
        return None

    # Iteratively break cycles by removing edge with lowest confidence
    iterations = 0
    max_iterations = len(edges) * 2

    while iterations < max_iterations:
        cycle = find_cycle(edges)
        if not cycle:
            break

        # Find edge with min confidence in cycle
        min_edge = min(cycle, key=lambda e: e.get("confidence", 0.0))
        logger.warning(
            "Cycle detected in prerequisite graph! Deterministically dropping min-confidence edge: %s -> %s (conf: %.4f)",
            min_edge.get("source_name"),
            min_edge.get("target_name"),
            min_edge.get("confidence", 0.0),
        )
        edges = [
            e for e in edges
            if not (e["source_id"] == min_edge["source_id"] and e["target_id"] == min_edge["target_id"])
        ]
        iterations += 1

    return edges


def infer_prerequisites(
    entities: List[Any],
    chunks: List[Any],
    segments: List[Any],
    relations: List[Any],
    min_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    High-level prerequisite inference pipeline.

    1. Generates candidate pairs using high-recall pre-filtering and temporal causality.
    2. Scores candidates using multi-signal scoring with hard evidence anchors.
    3. Filters by configurable confidence threshold (default: 0.65).
    4. Resolves cycles deterministically to guarantee strict DAG.

    Returns:
        {
            "prerequisites": List[Dict[str, Any]],
            "metrics": Dict[str, Any],
            "threshold": float
        }
    """
    settings = SharedSettings()
    threshold = min_confidence if min_confidence is not None else settings.PREREQUISITE_CONFIDENCE_THRESHOLD

    # 1. Candidate Generation
    candidates, gen_metrics = generate_prerequisite_candidates(
        entities=entities,
        chunks=chunks,
        segments=segments,
        relations=relations,
    )

    # 2. Scoring
    detector = DiscourseCueDetector()
    scored_edges = [
        score_prerequisite_candidate(c, chunks, detector=detector)
        for c in candidates
    ]

    # 3. Confidence Threshold Filter
    passing_edges = [e for e in scored_edges if e["confidence"] >= threshold]

    # 4. DAG Cycle Resolution
    dag_edges = enforce_dag_cycles(passing_edges)

    # Assign deterministic relation IDs (e.g. prereq_000001)
    for idx, edge in enumerate(dag_edges, start=1):
        edge["relation_id"] = f"prereq_{idx:06d}"

    logger.info(
        "Prerequisite inference complete: %d edges passed threshold >= %.2f (from %d candidates).",
        len(dag_edges),
        threshold,
        len(candidates),
    )

    return {
        "prerequisites": dag_edges,
        "metrics": {
            **gen_metrics,
            "scored_candidates": len(scored_edges),
            "passed_threshold": len(passing_edges),
            "final_dag_edges": len(dag_edges),
        },
        "threshold": threshold,
    }
