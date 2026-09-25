# evaluation/tests/test_routing_evaluation.py
"""Unit tests for query routing evaluation."""

from pathlib import Path
import pytest

from evaluation.evaluate_routing import evaluate_routing, generate_routing_markdown_report


def test_evaluate_routing_on_dataset():
    dataset_path = Path("evaluation/datasets/cs162_lecture1_routing_12.json")
    assert dataset_path.exists(), "Routing dataset must exist"

    results = evaluate_routing(dataset_path)

    assert results["total_samples"] == 12
    assert results["correct_samples"] == 12
    assert results["overall_accuracy"] == 1.0

    per_class = results["per_class_metrics"]
    assert "vector_only" in per_class
    assert "graph_only" in per_class
    assert "graph+vector" in per_class

    assert per_class["vector_only"]["total"] == 5
    assert per_class["graph_only"]["total"] == 5
    assert per_class["graph+vector"]["total"] == 2

    cm = results["confusion_matrix"]
    assert cm["vector_only"]["vector_only"] == 5
    assert cm["graph_only"]["graph_only"] == 5
    assert cm["graph+vector"]["graph+vector"] == 2


def test_generate_routing_markdown_report():
    dataset_path = Path("evaluation/datasets/cs162_lecture1_routing_12.json")
    results = evaluate_routing(dataset_path)
    md = generate_routing_markdown_report(results)

    assert "# LectureMIND — Query Routing Evaluation Report" in md
    assert "Confusion Matrix" in md
    assert "Per-Class Route Accuracy" in md
    assert "Benchmark Scope & Statistical Limitations" in md
