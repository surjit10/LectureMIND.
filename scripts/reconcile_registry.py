#!/usr/bin/env python3
"""Safely reconcile registry metadata without deleting packages or DB data.

Dry-run is the default. ``--apply`` only records hashes for valid extracted
directories and marks unusable/changed entries FAILED; it never extracts ZIPs,
deletes files, or modifies Qdrant/Neo4j.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from local.storage.lecture_registry import LectureRegistry, compute_package_hash  # noqa: E402


def reconcile(registry: LectureRegistry, apply: bool = False) -> dict:
    actions = []
    for lecture_id, entry in registry._registry.get("lectures", {}).items():
        package_path = Path(entry.get("package_path", ""))
        expected = entry.get("package_sha256", "")

        if not package_path.is_dir():
            reason = "path_missing" if not package_path.exists() else "path_not_directory"
            actions.append({
                "lecture_id": lecture_id,
                "action": "mark_failed",
                "reason": reason,
                "path": str(package_path),
            })
            if apply:
                registry.update_status(lecture_id, "FAILED")
            continue

        actual = compute_package_hash(package_path)
        if not expected and actual:
            actions.append({
                "lecture_id": lecture_id,
                "action": "record_content_hash",
                "sha256": actual,
            })
            if apply:
                registry.update_metadata(lecture_id, package_sha256=actual)
        elif expected and actual != expected:
            actions.append({
                "lecture_id": lecture_id,
                "action": "mark_failed",
                "reason": "content_hash_mismatch",
                "expected": expected,
                "actual": actual,
            })
            if apply:
                registry.update_status(lecture_id, "FAILED")

    return {
        "mode": "apply" if apply else "dry-run",
        "actions": actions,
        "action_count": len(actions),
        "destructive_actions": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile LectureMIND registry metadata")
    parser.add_argument("--registry-file", default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    registry = LectureRegistry(args.registry_file) if args.registry_file else LectureRegistry()
    report = reconcile(registry, apply=args.apply)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
