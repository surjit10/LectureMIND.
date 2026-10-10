"""Validate benchmark structure and release-readiness without inventing labels.

The smoke profile checks data integrity.  The release profile additionally
enforces minimum diversity so a single-lecture, single-route benchmark cannot
be presented as broad production evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from evaluation.dataset_loader import DatasetLoader


def validate_benchmark(path: str | Path, profile: str = "smoke") -> Dict[str, Any]:
    path = Path(path)
    samples = DatasetLoader(str(path)).load()
    issues: List[str] = []
    warnings: List[str] = []

    queries = [" ".join(sample.query.lower().split()) for sample in samples]
    duplicate_queries = sorted(q for q, count in Counter(queries).items() if count > 1)
    if duplicate_queries:
        issues.append(f"duplicate normalized queries: {len(duplicate_queries)}")

    for index, sample in enumerate(samples):
        if not sample.query.strip() or not sample.ground_truth_answer.strip():
            issues.append(f"sample {index} has an empty query or ground-truth answer")
        if not sample.expected_chunk_ids and sample.answerable:
            issues.append(f"sample {index} has no expected_chunk_ids")

    lecture_counts = Counter(sample.lecture_id for sample in samples)
    route_counts = Counter(sample.expected_route for sample in samples)
    difficulty_counts = Counter(sample.difficulty.lower() for sample in samples)
    type_counts = Counter(sample.question_type.lower() for sample in samples)
    visual_count = sum(bool(sample.need_visual) for sample in samples)

    if profile == "release":
        if len(samples) < 100:
            issues.append("release benchmark requires at least 100 labeled samples")
        if len(lecture_counts) < 2:
            issues.append("release benchmark requires at least 2 lectures")
        if len(route_counts) < 2:
            issues.append("release benchmark requires at least 2 expected routes")
        if difficulty_counts.get("hard", 0) < max(5, round(len(samples) * 0.10)):
            issues.append("release benchmark requires at least 10% hard questions (minimum 5)")
        if visual_count < max(5, round(len(samples) * 0.10)):
            issues.append("release benchmark requires at least 10% visual questions (minimum 5)")
        sparse_types = sorted(name for name, count in type_counts.items() if count < 5)
        if sparse_types:
            issues.append(
                "each represented question type needs at least 5 samples; sparse: "
                + ", ".join(sparse_types)
            )
        review_statuses = {sample.review_status for sample in samples}
        # Preserve compatibility with legacy datasets that predate review
        # metadata (unknown is a warning). Explicitly marked generated or
        # historical records must not pass as release-grade.
        if review_statuses == {"unknown"}:
            warnings.append("review_status is absent; human review cannot be verified")
        elif review_statuses != {"human_reviewed"}:
            issues.append(
                "release benchmark requires every sample to have review_status=human_reviewed; "
                f"found: {sorted(review_statuses)}"
            )
    elif profile != "smoke":
        raise ValueError("profile must be 'smoke' or 'release'")

    if len(lecture_counts) == 1:
        warnings.append("all samples come from one lecture")
    if len(route_counts) == 1:
        warnings.append("all samples use one expected route")

    return {
        "status": "PASS" if not issues else "FAIL",
        "profile": profile,
        "dataset": str(path),
        "sample_count": len(samples),
        "lecture_distribution": dict(sorted(lecture_counts.items())),
        "route_distribution": dict(sorted(route_counts.items())),
        "difficulty_distribution": dict(sorted(difficulty_counts.items())),
        "question_type_distribution": dict(sorted(type_counts.items())),
        "visual_count": visual_count,
        "issues": issues,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a LectureMIND benchmark")
    parser.add_argument("dataset")
    parser.add_argument("--profile", choices=("smoke", "release"), default="smoke")
    args = parser.parse_args()
    report = validate_benchmark(args.dataset, args.profile)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
