# evaluation/tests/test_dashboard.py
# Feature 4 — Benchmark Dashboard. Verifies the generator renders existing
# outputs into a self-contained HTML page and never touches the live eval
# stack (pure reads from fixture files).

import json
from pathlib import Path

from evaluation.dashboard.dashboard_generator import (
    generate_dashboard,
    load_benchmark_reports,
    load_load_test,
    load_ragas,
    svg_bars,
)


def _write_fixtures(tmp_path: Path) -> tuple:
    outputs = tmp_path / "outputs"
    reports = tmp_path / "reports"
    outputs.mkdir()
    reports.mkdir()

    (outputs / "evaluation_report_20260101_000000.json").write_text(
        json.dumps(
            {
                "timestamp": "20260101_000000",
                "total_samples": 10,
                "aggregated_metrics": {
                    "mrr": 0.85,
                    "recall_at_5": 0.9,
                    "ndcg_at_5": 0.88,
                    "routing_accuracy": 1.0,
                    "citation_coverage": 0.95,
                    "total_latency": 1.5,
                },
                "results": [],
            }
        ),
        encoding="utf-8",
    )

    # Newer report format: aggregated metrics are {mean, p95, min, max, n}.
    (outputs / "evaluation_report_20260202_000000.json").write_text(
        json.dumps(
            {
                "timestamp": "20260202_000000",
                "total_samples": 5,
                "aggregated_metrics": {
                    "mrr": {"mean": 0.71, "p95": 1.0, "min": 0.0, "max": 1.0, "n": 5},
                    "routing_accuracy": {"mean": 1.0, "p95": 1.0, "min": 1.0, "max": 1.0, "n": 5},
                },
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    (reports / "ragas_report.json").write_text(
        json.dumps({"Faithfulness": 0.92, "Answer Relevancy": 0.88, "Context Precision": 0.8}),
        encoding="utf-8",
    )
    (reports / "load_test_report.csv").write_text(
        "users,total_requests,errors,avg_ttft,avg_latency,p95_latency,throughput_rps\n"
        "100,100,0,0.2,1.1,1.8,90.0\n",
        encoding="utf-8",
    )
    return outputs, reports


class TestLoaders:
    def test_load_benchmark_reports(self, tmp_path):
        outputs, _ = _write_fixtures(tmp_path)
        reports = load_benchmark_reports(outputs)
        assert len(reports) == 2
        assert reports[0]["metrics"]["mrr"] == 0.85

    def test_load_new_nested_metric_format(self, tmp_path):
        from evaluation.dashboard.dashboard_generator import _metric_value
        assert _metric_value({"mean": 0.71, "p95": 1.0}) == 0.71
        assert _metric_value(0.85) == 0.85

    def test_dashboard_renders_nested_format(self, tmp_path):
        from evaluation.dashboard.dashboard_generator import _metric_bars
        metrics = {"mrr": {"mean": 0.71, "p95": 1.0}, "recall_at_5": 0.9}
        bars = _metric_bars(metrics, [("mrr", "MRR"), ("recall_at_5", "Recall@5")])
        assert ("MRR", 0.71) in bars
        assert ("Recall@5", 0.9) in bars

    def test_load_ragas_missing_is_empty(self, tmp_path):
        assert load_ragas(tmp_path) == {}

    def test_load_load_test(self, tmp_path):
        _, reports = _write_fixtures(tmp_path)
        rows = load_load_test(reports)
        assert rows[0]["users"] == 100.0
        assert rows[0]["p95_latency"] == 1.8


class TestSvgBars:
    def test_renders_label_and_value(self):
        svg = svg_bars([("MRR", 0.85), ("Recall@5", 0.9)])
        assert "<svg" in svg
        assert "MRR" in svg
        assert "0.85" in svg

    def test_empty_returns_no_data(self):
        assert "No data" in svg_bars([])


class TestGenerateDashboard:
    def test_generates_self_contained_html(self, tmp_path):
        outputs, reports = _write_fixtures(tmp_path)
        out = tmp_path / "index.html"
        generate_dashboard(outputs_dir=outputs, reports_dir=reports, out_html=out)
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "LectureMIND Benchmark Dashboard" in content
        assert "MRR" in content
        assert "Faithfulness" in content
        assert "Load test" in content
        # Must be self-contained: inline <style>, no external asset references.
        assert "<style>" in content
        assert "<link" not in content
        assert 'src="http' not in content and "src='http" not in content

    def test_empty_outputs_renders_no_data(self, tmp_path):
        out = tmp_path / "empty.html"
        generate_dashboard(outputs_dir=tmp_path, reports_dir=tmp_path, out_html=out)
        content = out.read_text(encoding="utf-8")
        assert "No benchmark reports found" in content
