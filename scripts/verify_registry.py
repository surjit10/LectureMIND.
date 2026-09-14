#!/usr/bin/env python3
"""Verify package-registry integrity: referenced packages exist and match their
recorded SHA-256 content hashes.

Usage:
    python scripts/verify_registry.py            # human-readable summary
    python scripts/verify_registry.py --json     # machine-readable report

Exit code 0 = all verified; 1 = one or more problems found.

What this prevents (audit finding, 2026-09-14):
    - benchmark/evaluation referencing packages that no longer exist
      (the `lecture_cs162_v17` incident), and
    - silent content drift between the zip a metric was computed on and the
      package currently on disk.

Run this in CI and before any benchmark/audit to guarantee the evaluation
artifacts and the data they reference are in sync.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from local.storage.lecture_registry import (  # noqa: E402
    LectureRegistry,
    compute_package_zip_hash,
)


def _hash_recorded_path(entry: dict) -> tuple[str | None, str | None]:
    """Hash the artifact a registry entry points at.

    Entries may point at a ZIP artifact (hash = file SHA-256) or a package
    directory (hash = deterministic dir digest). Returns (hash, error).
    """
    from local.storage.lecture_registry import _hash_package_dir

    p = Path(entry.get("package_path", ""))
    if not p.exists():
        return None, f"missing on disk: {p}"
    if p.is_file():
        return compute_package_zip_hash(p), None
    return _hash_package_dir(p), None


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify LectureMIND package registry integrity")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON report")
    parser.add_argument(
        "--registry-file",
        default=None,
        help="override registry file path (default: data/lecture_registry.json)",
    )
    args = parser.parse_args()

    registry = (
        LectureRegistry(args.registry_file) if args.registry_file else LectureRegistry()
    )
    lectures = registry._registry.get("lectures", {})

    problems: list[dict] = []
    verified: list[str] = []
    unhashed: list[str] = []

    for lid, entry in lectures.items():
        if not entry.get("package_sha256"):
            unhashed.append(lid)
            continue
        actual, error = _hash_recorded_path(entry)
        if error is not None:
            problems.append({"lecture_id": lid, "issue": error})
            continue
        if actual != entry["package_sha256"]:
            problems.append(
                {
                    "lecture_id": lid,
                    "issue": "content hash mismatch",
                    "expected": entry["package_sha256"],
                    "actual": actual,
                }
            )
            continue
        verified.append(lid)

    report = {
        "total_registered": len(lectures),
        "verified": sorted(verified),
        "problems": problems,
        "no_hash_recorded": sorted(unhashed),
        "status": "OK" if not problems else "FAILED",
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Registry integrity: {report['status']}")
        print(f"  registered: {report['total_registered']}")
        print(f"  verified:   {len(verified)}")
        if unhashed:
            print(f"  no hash recorded (legacy entries): {len(unhashed)}")
        for p in problems:
            print(f"  PROBLEM: {p['lecture_id']}: {p['issue']}")
            if "expected" in p:
                print(f"    expected {p['expected'][:16]}... got {p['actual'][:16]}...")

    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
