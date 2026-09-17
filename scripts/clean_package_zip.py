#!/usr/bin/env python3
"""Clean an existing knowledge package ZIP without re-running the GPU pipeline.

Applies the post-audit-fix quality gates to an OLD package (generated before
the evidence rule / fragment filter existed):

  1. Entities: drops names flagged FRAGMENT or EXCESSIVE_LENGTH by the KG
     auditor, plus anything caught by the ingestion gate is_fragment_entity().
  2. Relations: drops edges whose evidence tier is CO_OCCURRENCE_ONLY or
     NO_EVIDENCE — exactly the edges the new evidence rule would never have
     saved (no verifiable quote in the lecture text).
  3. Drops now-dangling relations, renumbers relation IDs, and updates the
     manifest counts so the package stays internally consistent.

The original ZIP is never modified; a cleaned copy is written to --out.

Usage:
    python scripts/clean_package_zip.py \
        --package 0-output/<lecture>_knowledge_package.zip \
        --out outputs/<lecture>_cleaned.zip \
        [--keep-cooccurrence]
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud.extraction.entity_extractor import is_fragment_entity  # noqa: E402
from evaluation.knowledge_graph.entity_auditor import audit_entities  # noqa: E402
from evaluation.knowledge_graph.relation_auditor import audit_relations  # noqa: E402

# Evidence tiers that the post-fix ingestion pipeline would have rejected.
DROP_TIERS = {"CO_OCCURRENCE_ONLY", "NO_EVIDENCE"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Strip audit-flagged junk from a knowledge package ZIP")
    ap.add_argument("--package", required=True, help="path to the original knowledge package ZIP")
    ap.add_argument("--out", required=True, help="path for the cleaned ZIP copy")
    ap.add_argument(
        "--keep-cooccurrence",
        action="store_true",
        help="only drop NO_EVIDENCE edges; keep CO_OCCURRENCE_ONLY edges",
    )
    args = ap.parse_args()

    drop_tiers = DROP_TIERS - ({"CO_OCCURRENCE_ONLY"} if args.keep_cooccurrence else set())

    with zipfile.ZipFile(args.package) as zf:
        names = zf.namelist()
        data = {n: zf.read(n) for n in names}

    entities = json.loads(data["entities.json"])
    relations = json.loads(data["relations.json"])
    chunks = json.loads(data["multimodal_chunks.json"])

    # ── 1. Decide which entities to drop ─────────────────────────────────
    ea = audit_entities(entities, relations)
    drop_entity_ids = {
        i["entity_id"] for i in ea.issues if i["issue_type"] in {"FRAGMENT", "EXCESSIVE_LENGTH"}
    }
    # Belt and braces: also run the ingestion gate itself.
    for e in entities:
        if is_fragment_entity(e.get("name", "")):
            drop_entity_ids.add(e["entity_id"])

    kept_entities = [e for e in entities if e["entity_id"] not in drop_entity_ids]
    kept_ids = {e["entity_id"] for e in kept_entities}

    # ── 2. Decide which relations to drop ────────────────────────────────
    ra = audit_relations(relations, entities, chunks)
    tier_by_rid = {d["relation_id"]: d["evidence_tier"] for d in ra.relation_details}

    kept_relations = []
    for r in relations:
        if tier_by_rid.get(r["relation_id"]) in drop_tiers:
            continue
        if r["source_entity_id"] not in kept_ids or r["target_entity_id"] not in kept_ids:
            continue  # dangling after entity removal
        kept_relations.append(r)
    for idx, r in enumerate(kept_relations, start=1):
        r["relation_id"] = f"rel_{idx:06d}"

    # ── 3. Write the cleaned copy ────────────────────────────────────────
    manifest = json.loads(data["manifest.json"])
    manifest["entity_count"] = len(kept_entities)
    manifest["relation_count"] = len(kept_relations)
    data["manifest.json"] = json.dumps(manifest, indent=2).encode("utf-8")
    data["entities.json"] = json.dumps(kept_entities, indent=2).encode("utf-8")
    data["relations.json"] = json.dumps(kept_relations, indent=2).encode("utf-8")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for n in names:
            zf.writestr(n, data[n])

    print(f"entities : {len(entities)} -> {len(kept_entities)} (dropped {len(entities) - len(kept_entities)})")
    print(f"relations: {len(relations)} -> {len(kept_relations)} (dropped {len(relations) - len(kept_relations)})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
