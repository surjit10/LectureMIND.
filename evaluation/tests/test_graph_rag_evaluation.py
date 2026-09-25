# evaluation/tests/test_graph_rag_evaluation.py
"""Unit tests for GraphRAG and multi-hop benchmark evaluation."""

from pathlib import Path
import pytest

from evaluation.evaluate_graph_rag import evaluate_graph_benchmark, generate_graph_markdown_report


def test_evaluate_graph_benchmark():
    dataset_path = Path("evaluation/datasets/cs162_lecture1_graph_qa.json")
    pkg_dir = Path("data/packages/lecture_092f861b")
    if not (pkg_dir / "entities.json").exists():
        pytest.skip("CS162 lecture package not present (gitignored in CI)")
    assert dataset_path.exists(), "Graph benchmark dataset must exist"

    results = evaluate_graph_benchmark(dataset_path)

    assert results["total_queries"] == 20
    st = results["structural_traversal"]
    assert st["total_queries"] == 20
    assert "1_hop" in st["by_hop"]
    assert "2_hop" in st["by_hop"]
    assert "3_hop" in st["by_hop"]

    cr = results["downstream_chunk_retrieval"]
    assert "bm25_metrics" in cr
    assert "graph_metrics" in cr
    assert "hybrid_metrics" in cr


def test_generate_graph_markdown_report():
    import json
    report_path = Path("evaluation/outputs/graph_evaluation_report.json")
    assert report_path.exists(), "Graph evaluation report must exist"
    with open(report_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    md = generate_graph_markdown_report(results)

    assert "# LectureMIND — GraphRAG & Multi-Hop Traversal Evaluation Report" in md
    assert "Structural Graph Traversal Accuracy" in md
    assert "Downstream Chunk Retrieval Comparison on Graph Queries" in md
    assert "Hop-by-Hop Chunk Retrieval Breakdown" in md


