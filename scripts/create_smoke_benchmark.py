"""Create a small, balanced smoke benchmark from the reviewed-source draft.

This selects representative records; it does not change labels or optimize
selection for a target score. The result remains provisional until human review.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _take(items, predicate, count, used):
    out = []
    for item in items:
        key = (item["lecture_id"], item["query"])
        if key in used or not predicate(item):
            continue
        used.add(key)
        out.append(item)
        if len(out) == count:
            break
    return out


def main() -> int:
    source = ROOT / "evaluation/datasets/multilecture_groq_eval_draft.json"
    records = json.loads(source.read_text())
    by_lecture = {}
    for item in records:
        by_lecture.setdefault(item["lecture_id"], []).append(item)

    selected = []
    used = set()
    # OS: 10 source-grounded Groq items + 3 graph positives + 2 negatives.
    os_items = by_lecture["lecture_092f861b"]
    selected += _take(os_items, lambda x: x["review_status"] == "auto_generated_pending_human_review" and x["answerable"] and x["question_type"] != "visual", 6, used)
    selected += _take(os_items, lambda x: x["review_status"] == "auto_generated_pending_human_review" and x["answerable"] and x["question_type"] == "visual", 4, used)
    selected += _take(os_items, lambda x: x["question_type"] == "graph" and x["answerable"], 3, used)
    selected += _take(os_items, lambda x: x["question_type"] == "unanswerable" and not x["answerable"], 2, used)

    # Deep learning and transformer lectures: 10 factual/conceptual + 5 visual.
    for lecture_id in ("lecture_58c030b4", "lecture_ddf8d2b5"):
        items = by_lecture[lecture_id]
        visual_target = min(5, sum(bool(x["need_visual"]) for x in items))
        selected += _take(items, lambda x: x["answerable"] and not x["need_visual"], 15 - visual_target, used)
        selected += _take(items, lambda x: x["answerable"] and x["need_visual"], visual_target, used)

    if len(selected) != 45:
        raise RuntimeError(f"Expected 45 smoke records, selected {len(selected)}")

    out = ROOT / "evaluation/datasets/multilecture_smoke_45.json"
    out.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "dataset_name": "LectureMind balanced smoke benchmark",
        "record_count": len(selected),
        "per_lecture": {lid: sum(x["lecture_id"] == lid for x in selected) for lid in by_lecture if any(x["lecture_id"] == lid for x in selected)},
        "status": "provisional_not_human_reviewed",
        "selection_policy": "fixed category-balanced smoke sample; no score optimization",
        "source": str(source),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
