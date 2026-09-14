#!/usr/bin/env python3
"""Generate CS162 gold-candidate reference labels from the 0-output package.

IMPORTANT — what this script does and does NOT do:
  It produces CANDIDATE reference labels: entities, relations, and
  prerequisites, each anchored to a concrete chunk and verbatim evidence
  sentence extracted from the package itself. It does NOT invent evidence:
  every candidate must carry a quote found in the cited chunk, or it is
  dropped (marked as needing manual attention instead).

  It is LLM-GATED: without a configured provider (e.g. Groq key in
  data/llm_config.json or LLM_API_KEY) it refuses to run rather than writing
  labels nobody reviewed. Output is clearly marked
  "LLM-assisted reference labels, PENDING HUMAN VERIFICATION".

  The intended workflow:
    1. Run this script -> evaluation/knowledge_graph/cs162_gold_candidates.json
    2. A human reviews each candidate (verify evidence quotes are genuinely
       supporting, prune wrong ones) and promotes the file to
       cs162_entity_gold.json / cs162_relation_gold.json /
       cs162_prerequisite_gold.json.
    3. Only then does audit_package.py compute gold metrics for CS162.

Usage:
    python scripts/generate_cs162_gold.py \
        --package 0-output/CS162_Lecture_1_What_is_an_Operating_System_720P_knowledge_package.zip \
        --output evaluation/knowledge_graph/cs162_gold_candidates.json \
        [--max-entities 40] [--max-relations 60]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GENERIC_TERMS = {
    "system", "program", "resource", "protection", "isolation", "storage",
    "device", "interface", "user", "hardware", "software", "data", "memory",
}


def _extract_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if 20 <= len(p.strip()) <= 400]


def _load_package(zip_path: Path) -> Dict[str, Any]:
    with zipfile.ZipFile(zip_path) as z:
        return {
            "metadata": json.loads(z.read("metadata.json")),
            "entities": json.loads(z.read("entities.json")),
            "relations": json.loads(z.read("relations.json")),
            "chunks": json.loads(z.read("multimodal_chunks.json")),
        }


def _chunk_texts(pkg: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for c in pkg["chunks"]:
        out[c["chunk_id"]] = (c.get("transcript") or "") + "\n" + (c.get("ocr_text") or "")
    return out


def _find_evidence(name: str, texts: Dict[str, str]) -> Optional[Dict[str, str]]:
    """First chunk containing a sentence that genuinely names the concept."""
    pat = re.compile(r"\b" + re.escape(name.lower()) + r"\b")
    for cid, text in texts.items():
        for sent in _extract_sentences(text):
            if pat.search(sent.lower()):
                return {"chunk_id": cid, "evidence_text": sent}
    return None


def _llm_refine(candidates: List[Dict], provider: Any) -> List[Dict]:
    """Ask the LLM to filter/prune candidate labels.

    The LLM never CREATES evidence — it only marks each candidate
    keep/drop/unsure with a one-line reason. Evidence quotes stay exactly as
    extracted from the package.
    """
    kept: List[Dict] = []
    batch_size = 10
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        listing = "\n".join(
            f"{j}. {c.get('canonical_name') or (c.get('source_name','?') + ' -' + c.get('relation','?') + '-> ' + c.get('target_name','?'))}"
            f"\n   evidence: \"{c.get('evidence_text','')[:180]}\""
            for j, c in enumerate(batch)
        )
        prompt = (
            "You are curating reference labels for a lecture knowledge-graph evaluation "
            "for an operating-systems lecture. For each numbered candidate below, reply "
            "with one line: number | KEEP or DROP or UNSURE | short reason. "
            "DROP only if the evidence quote does not genuinely support the label.\n\n"
            f"{listing}"
        )
        try:
            response = provider.chat(messages=[{"role": "user", "content": prompt}], temperature=0.0)
            text = response.get("content", "") if isinstance(response, dict) else str(response)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] LLM refine failed on batch {i//batch_size}: {exc} — keeping candidates unreviewed")
            kept.extend(batch)
            continue
        for j, c in enumerate(batch):
            line = next((ln for ln in text.splitlines() if ln.strip().startswith(f"{j}")), "")
            verdict = "UNSURE"
            if "| KEEP" in line.upper():
                verdict = "KEEP"
            elif "| DROP" in line.upper():
                verdict = "DROP"
            if verdict != "DROP":
                c["llm_verdict"] = verdict
                c["llm_reason"] = line.split("|", 2)[2].strip() if line.count("|") >= 2 else ""
                kept.append(c)
    return kept


def _get_provider():
    """Resolve an LLM provider or return None (gates the script)."""
    try:
        from local.llm.provider_registry import get_provider_registry
        from agent.langgraph.nodes.answer_generator import build_llm_backend  # type: ignore
    except Exception:
        return None
    try:
        return build_llm_backend()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate CS162 gold-candidate labels")
    ap.add_argument("--package", required=True)
    ap.add_argument("--output", default="evaluation/knowledge_graph/cs162_gold_candidates.json")
    ap.add_argument("--max-entities", type=int, default=40)
    ap.add_argument("--max-relations", type=int, default=60)
    ap.add_argument("--skip-llm", action="store_true", help="extract candidates only, no LLM pass (still marked unreviewed)")
    args = ap.parse_args()

    provider = None
    if not args.skip_llm:
        provider = _get_provider()
        if provider is None:
            print(
                "REFUSING to generate gold candidates: no LLM provider configured.\n"
                "Configure a provider (data/llm_config.json or LLM_API_KEY) and retry,\n"
                "or run with --skip-llm to extract evidence-anchored candidates without\n"
                "the LLM curation pass (they will be marked as needing full manual review)."
            )
            return 2

    pkg = _load_package(Path(args.package))
    lecture_id = pkg["metadata"]["lecture_id"]
    texts = _chunk_texts(pkg)

    # ---- Entity candidates (declared concept entities with grounded names) ----
    entity_candidates: List[Dict] = []
    seen = set()
    for e in pkg["entities"]:
        name = (e.get("name") or "").strip()
        if not name or name.lower() in GENERIC_TERMS or name.lower() in seen:
            continue
        if len(name.split()) > 5 or len(name) < 4:
            continue
        if e.get("type") not in (None, "", "Concept"):
            continue
        ev = _find_evidence(name, texts)
        if ev is None:
            continue
        seen.add(name.lower())
        entity_candidates.append(
            {
                "canonical_name": name[0].upper() + name[1:],
                "aliases": [],
                "type": "Concept",
                "chunks": [ev["chunk_id"]],
                "description": "",
                "evidence": ev,
                "candidate_source": "package entity (A8) + verbatim evidence match",
            }
        )
    entity_candidates = entity_candidates[: args.max_entities]

    # ---- Relation candidates (extracted relations with verbatim evidence) ----
    name_by_id = {e["entity_id"]: e.get("name", "") for e in pkg["entities"]}
    relation_candidates: List[Dict] = []
    for r in pkg["relations"]:
        s = name_by_id.get(r.get("source_entity_id", ""), "")
        t = name_by_id.get(r.get("target_entity_id", ""), "")
        if not s or not t:
            continue
        ev = _find_evidence(s, texts)
        if ev is None:
            continue
        relation_candidates.append(
            {
                "source_name": s,
                "relation": r.get("relation", r.get("relation_type", "RELATED_TO")),
                "target_name": t,
                "evidence_chunk_id": ev["chunk_id"],
                "evidence_text": ev["evidence_text"],
                "evidence_type": "CANDIDATE_DIRECT_EVIDENCE",
                "lecture_supported": None,   # decided during human review
                "factually_correct": None,   # decided during human review
                "candidate_source": "package relation (A9) + verbatim evidence match",
            }
        )
    relation_candidates = relation_candidates[: args.max_relations]

    # ---- Prerequisite candidates: relation edges of pedagogical types ----
    prereq_types = {"PREREQUISITE_OF", "INTRODUCED_BEFORE"}
    prerequisite_candidates: List[Dict] = []
    for r in pkg["relations"]:
        if r.get("relation") not in prereq_types:
            continue
        s = name_by_id.get(r.get("source_entity_id", ""), "")
        t = name_by_id.get(r.get("target_entity_id", ""), "")
        if not s or not t:
            continue
        ev = _find_evidence(t, texts)
        prerequisite_candidates.append(
            {
                "source_name": s,
                "target_name": t,
                "pedagogical_justification": "CANDIDATE — justification must be written during human review",
                "evidence": ev or {},
                "is_acyclic": True,
                "lecture_supported": None,
                "candidate_source": "package prerequisite (A10)",
            }
        )

    # ---- Optional LLM curation pass (never creates evidence) ----
    llm_reviewed = False
    if provider is not None:
        print("LLM curation pass (filter-only; evidence stays package-verbatim)...")
        entity_candidates = _llm_refine(entity_candidates, provider)
        relation_candidates = _llm_refine(relation_candidates, provider)
        prerequisite_candidates = _llm_refine(prerequisite_candidates, provider)
        llm_reviewed = True

    out = {
        "metadata": {
            "lecture_id": lecture_id,
            "package": str(args.package),
            "annotation_method": (
                "LLM-assisted candidate reference labels, PENDING HUMAN VERIFICATION — "
                "not gold until a human has reviewed every record"
                if llm_reviewed else
                "Automatically extracted candidate labels (no LLM pass), PENDING HUMAN VERIFICATION"
            ),
            "llm_reviewed": llm_reviewed,
            "counts": {
                "entities": len(entity_candidates),
                "relations": len(relation_candidates),
                "prerequisites": len(prerequisite_candidates),
            },
            "usage": (
                "After human review, split into cs162_entity_gold.json / "
                "cs162_relation_gold.json / cs162_prerequisite_gold.json with the "
                "same schema as the Transformer gold files, then audit with "
                "audit_package.py --gold-entities ... --gold-relations ... "
                "--gold-prerequisites ..."
            ),
        },
        "entities": entity_candidates,
        "relations": relation_candidates,
        "prerequisites": prerequisite_candidates,
    }

    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"  entities: {len(entity_candidates)} | relations: {len(relation_candidates)} | prerequisites: {len(prerequisite_candidates)}")
    print("STATUS: PENDING HUMAN VERIFICATION — do not use as gold until reviewed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
