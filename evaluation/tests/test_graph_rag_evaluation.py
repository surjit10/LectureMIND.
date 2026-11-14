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




def test_heldout_markdown_is_computed_from_data():
    """Regression: per-hop Hit@5, deltas and narratives used to be hard-coded strings."""
    import copy
    import json
    from evaluation.run_heldout_graph_evaluation import generate_markdown_report

    report_path = Path("evaluation/outputs/graph_heldout_evaluation.json")
    if not report_path.exists():
        pytest.skip("held-out result file not present")
    data = json.loads(report_path.read_text(encoding="utf-8"))

    def hop_hit5(d, hop):
        items = [x for x in d["detailed_results"] if x["answerable"] and x["hop_count"] == hop]
        return sum(x["chunk_retrieval"]["hit_at_5"] for x in items), len(items)

    md = generate_markdown_report(data)
    for hop in (1, 2, 3):
        hits, total = hop_hit5(data, hop)
        row = next(line for line in md.splitlines() if line.startswith(f"| {hop}-hop |"))
        assert f"| {hits}/{total} (" in row

    # Flipping one result must change the rendered table (no frozen literals).
    mutated = copy.deepcopy(data)
    mutated["summary"].pop("downstream_hit5_by_hop", None)
    first_1hop = next(x for x in mutated["detailed_results"] if x["answerable"] and x["hop_count"] == 1)
    first_1hop["chunk_retrieval"]["hit_at_5"] = 1 - first_1hop["chunk_retrieval"]["hit_at_5"]
    hits, total = hop_hit5(mutated, 1)
    row = next(line for line in generate_markdown_report(mutated).splitlines() if line.startswith("| 1-hop |"))
    assert f"| {hits}/{total} (" in row

    for stale in ("+20% over BM25", "AUTHENTIC MEASUREMENT", "END-TO-END VERIFIED",
                  "0 false edges", "LLM correctly refused", "MRR@5 |"):
        assert stale not in md
