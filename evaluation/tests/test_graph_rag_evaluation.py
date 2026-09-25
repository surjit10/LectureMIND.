# evaluation/tests/test_graph_rag_evaluation.py
"""Unit tests for GraphRAG and multi-hop benchmark evaluation."""

from pathlib import Path
import pytest

from evaluation.evaluate_graph_rag import evaluate_graph_benchmark, generate_graph_markdown_report


def test_evaluate_graph_benchmark():
    dataset_path = Path("evaluation/datasets/cs162_lecture1_graph_qa.json")
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
    dataset_path = Path("evaluation/datasets/cs162_lecture1_graph_qa.json")
    results = evaluate_graph_benchmark(dataset_path)
    md = generate_graph_markdown_report(results)

    assert "# LectureMIND — GraphRAG & Multi-Hop Traversal Evaluation Report" in md
    assert "Structural Graph Traversal Accuracy" in md
    assert "Downstream Chunk Retrieval Comparison on Graph Queries" in md
    assert "Hop-by-Hop Chunk Retrieval Breakdown" in md
