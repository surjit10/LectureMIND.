# evaluation/tests/test_answer_error_analysis.py
"""Unit tests for Answer F1 error analysis."""

from pathlib import Path
import pytest

from evaluation.analyze_answer_errors import run_answer_error_analysis, generate_error_analysis_markdown


def test_run_answer_error_analysis():
    report_file = Path("evaluation/outputs/evaluation_report_20260811_105621.json")
    dataset_file = Path("evaluation/datasets/cs162_lecture1_qa_50.json")

    assert report_file.exists()
    assert dataset_file.exists()

    analysis = run_answer_error_analysis(report_file, dataset_file)

    assert analysis["total_analyzed"] == 50
    assert analysis["mean_answer_f1"] > 0.40

    counts = analysis["category_counts"]
    total_categorized = sum(counts.values())
    assert total_categorized == 50
    assert counts["retrieval_failure"] == 1


def test_generate_error_analysis_markdown():
    report_file = Path("evaluation/outputs/evaluation_report_20260811_105621.json")
    dataset_file = Path("evaluation/datasets/cs162_lecture1_qa_50.json")

    analysis = run_answer_error_analysis(report_file, dataset_file)
    md = generate_error_analysis_markdown(analysis)

    assert "# LectureMIND — Answer F1 Deep Error Analysis" in md
    assert "Error Category Distribution" in md
    assert "Representative Case Studies" in md
