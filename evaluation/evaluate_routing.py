# evaluation/evaluate_routing.py
"""
Evaluation script for LectureMIND query routing.

Evaluates the QueryPlanner against the curated multi-route dataset:
    evaluation/datasets/cs162_lecture1_routing_12.json

Measures:
  - Overall route accuracy
  - Per-route accuracy (vector_only, graph_only, graph+vector)
  - Confusion matrix
  - Statistical context and limitations
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.dspy.planner import QueryPlanner
from schemas.enums import RetrievalRoute


def evaluate_routing(dataset_path: str | Path) -> Dict[str, Any]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Routing dataset not found: {dataset_file}")

    with open(dataset_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    planner = QueryPlanner()
    total = len(samples)
    correct = 0

    route_classes = [
        RetrievalRoute.vector_only.value,
        RetrievalRoute.graph_only.value,
        RetrievalRoute.graph_and_vector.value,
    ]

    confusion_matrix: Dict[str, Dict[str, int]] = {
        exp: {pred: 0 for pred in route_classes} for exp in route_classes
    }

    per_class_stats: Dict[str, Dict[str, int]] = {
        rc: {"total": 0, "correct": 0} for rc in route_classes
    }

    item_results: List[Dict[str, Any]] = []

    for item in samples:
        query = item["query"]
        expected = item["expected_route"]
        # Normalize alias "graph_and_vector" vs "graph+vector"
        if expected == "graph_and_vector":
            expected = RetrievalRoute.graph_and_vector.value

        plan = planner.plan_full(query)
        predicted = plan.retrieval_route.value

        is_match = (predicted == expected)
        if is_match:
            correct += 1

        confusion_matrix[expected][predicted] += 1
        per_class_stats[expected]["total"] += 1
        if is_match:
            per_class_stats[expected]["correct"] += 1

        item_results.append({
            "query": query,
            "expected_route": expected,
            "predicted_route": predicted,
            "match": is_match,
            "intent": plan.intent,
            "is_lecture_wide": plan.is_lecture_wide,
            "difficulty": item.get("difficulty", "medium"),
            "topic": item.get("topic", "general"),
        })

    overall_accuracy = correct / total if total > 0 else 0.0

    class_accuracies = {}
    for rc, stat in per_class_stats.items():
        class_accuracies[rc] = {
            "total": stat["total"],
            "correct": stat["correct"],
            "accuracy": round(stat["correct"] / stat["total"], 4) if stat["total"] > 0 else 0.0,
        }

    return {
        "dataset": str(dataset_file),
        "total_samples": total,
        "correct_samples": correct,
        "overall_accuracy": round(overall_accuracy, 4),
        "per_class_metrics": class_accuracies,
        "confusion_matrix": confusion_matrix,
        "samples": item_results,
        "statistical_context": {
            "n": total,
            "weight_per_sample_pct": round(100.0 / total, 2) if total > 0 else 0.0,
            "notes": (
                "Evaluated on cs162_lecture1_routing_12.json containing distinct "
                "vector_only (N=5), graph_only (N=5), and graph+vector (N=2) queries. "
                "Each sample represents 8.33 percentage points. Do not extrapolate "
                "generalization beyond the tested heuristic pattern coverage."
            ),
        },
    }


def generate_routing_markdown_report(report_data: Dict[str, Any]) -> str:
    n = report_data["total_samples"]
    acc = report_data["overall_accuracy"] * 100.0
    cm = report_data["confusion_matrix"]
    per_class = report_data["per_class_metrics"]

    md = [
        "# LectureMIND — Query Routing Evaluation Report",
        "",
        f"**Dataset:** `{report_data['dataset']}`  ",
        f"**Sample Size (N):** {n}  ",
        f"**Overall Accuracy:** {acc:.1f}% ({report_data['correct_samples']}/{n})  ",
        "",
        "## 1. Per-Class Route Accuracy",
        "",
        "| Route | Total | Correct | Accuracy |",
        "|---|---:|---:|---:|",
    ]

    for route, metrics in per_class.items():
        md.append(f"| `{route}` | {metrics['total']} | {metrics['correct']} | {metrics['accuracy']*100:.1f}% |")

    md.extend([
        "",
        "## 2. Confusion Matrix",
        "",
        "| Expected \\ Predicted | vector_only | graph_only | graph+vector |",
        "|---|---:|---:|---:|",
        f"| **vector_only** | {cm['vector_only']['vector_only']} | {cm['vector_only']['graph_only']} | {cm['vector_only']['graph+vector']} |",
        f"| **graph_only** | {cm['graph_only']['vector_only']} | {cm['graph_only']['graph_only']} | {cm['graph_only']['graph+vector']} |",
        f"| **graph+vector** | {cm['graph+vector']['vector_only']} | {cm['graph+vector']['graph_only']} | {cm['graph+vector']['graph+vector']} |",
        "",
        "## 3. Sample-by-Sample Details",
        "",
        "| # | Query | Expected | Predicted | Intent | Match |",
        "|---|---|---|---|---|:---:|",
    ])

    for i, s in enumerate(report_data["samples"], start=1):
        match_str = "PASS" if s["match"] else "FAIL"
        md.append(f"| {i} | {s['query'][:55]}... | `{s['expected_route']}` | `{s['predicted_route']}` | `{s['intent']}` | {match_str} |")

    md.extend([
        "",
        "## 4. Benchmark Scope & Statistical Limitations",
        "",
        f"- **Sample size:** N = {n} questions.",
        f"- **Resolution:** 1 sample = {report_data['statistical_context']['weight_per_sample_pct']}% of total accuracy.",
        "- **Coverage:** This dataset explicitly balances structural/graph queries (prerequisites, connections),",
        "  pure semantic explanation queries (vector_only), and composite relational+explanatory queries (graph+vector).",
        "- **Note on Headline Metrics:** The 50-QA benchmark (`cs162_lecture1_qa_50.json`) contains only `vector_only`",
        "  labels because it targets multimodal chunk retrieval. True multi-route classification performance",
        "  must be cited from this balanced 12-query routing benchmark, not the 50-QA benchmark.",
        "",
    ])

    return "\n".join(md)


def main():
    dataset_path = Path("evaluation/datasets/cs162_lecture1_routing_12.json")
    results = evaluate_routing(dataset_path)

    out_dir = Path("evaluation/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "routing_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_path = out_dir / "routing_evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_routing_markdown_report(results))

    print(f"Routing evaluation complete:")
    print(f"  Overall accuracy: {results['overall_accuracy']*100:.1f}% ({results['correct_samples']}/{results['total_samples']})")
    print(f"  Saved JSON: {json_path}")
    print(f"  Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
