"""Merge Groq-generated drafts with the existing CS162 set without hiding provenance."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    records = []
    for name in ("draft_os_20.json", "draft_dl_20.json", "draft_transformer_20.json"):
        records.extend(json.loads((ROOT / "evaluation/datasets" / name).read_text()))

    historical = json.loads((ROOT / "evaluation/datasets/cs162_lecture1_qa_50.json").read_text())
    for item in historical:
        item = dict(item)
        # The old dataset used a stale runtime ID, but its chunk IDs belong to
        # the registered CS162 package. Preserve it as historical/unreviewed.
        item["lecture_id"] = "lecture_092f861b"
        item["answerable"] = True
        item["review_status"] = "historical_unreviewed"
        item["source_evidence"] = {"chunk_ids": item.get("expected_chunk_ids", [])}
        records.append(item)

    # Add existing graph QA and held-out negative examples in the same schema.
    # They remain historical/unreviewed and are never treated as human gold.
    graph = json.loads((ROOT / "evaluation/datasets/cs162_lecture1_graph_qa.json").read_text())
    for item in graph:
        records.append({
            "lecture_id": "lecture_092f861b",
            "query": item["query"],
            "ground_truth_answer": item["ground_truth_answer"],
            "expected_chunk_ids": item.get("expected_chunk_ids", []),
            "expected_route": "graph_only",
            "question_type": "graph",
            "difficulty": item.get("difficulty", "medium"),
            "topic": item.get("topic", "graph"),
            "keywords": item.get("gold_entities", []),
            "need_visual": False,
            "answerable": True,
            "review_status": "historical_unreviewed",
            "source_evidence": {"chunk_ids": item.get("expected_chunk_ids", []), "gold_path": item.get("gold_path", [])},
        })
    heldout = json.loads((ROOT / "evaluation/datasets/cs162_lecture1_graph_qa_heldout.json").read_text())
    for item in heldout:
        records.append({
            "lecture_id": "lecture_092f861b",
            "query": item["question"],
            "ground_truth_answer": item["gold_answer"],
            "expected_chunk_ids": item.get("gold_source_chunk_ids", []),
            "expected_route": "graph_only",
            "question_type": "unanswerable" if not item.get("answerable", True) else "graph",
            "difficulty": "hard" if item.get("hop_count", 0) >= 3 else "medium",
            "topic": "graph",
            "keywords": item.get("gold_entities", []),
            "need_visual": False,
            "answerable": bool(item.get("answerable", True)),
            "review_status": "historical_unreviewed",
            "source_evidence": {"chunk_ids": item.get("gold_source_chunk_ids", []), "gold_path": item.get("gold_path", [])},
        })

    out = ROOT / "evaluation/datasets/multilecture_groq_eval_draft.json"
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "dataset_name": "LectureMind multi-lecture Groq evaluation draft",
        "label_status": "not_human_reviewed",
        "record_count": len(records),
        "sources": {
            "groq_generated": 60,
            "historical_unreviewed": len(historical),
            "graph_and_negative_additions": len(graph) + len(heldout),
        },
        "warning": "Do not use as release-grade ground truth until every item is independently human reviewed.",
    }
    (out.with_suffix(".manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
