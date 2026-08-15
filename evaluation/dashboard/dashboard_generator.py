# evaluation/dashboard/dashboard_generator.py
# Feature 4 — Benchmark Dashboard.
#
# Renders a self-contained HTML dashboard from EXISTING evaluation outputs.
# Never re-runs benchmarks, RAGAS, or load tests — it only reads:
#   - evaluation/outputs/evaluation_report_*.json   (benchmark_runner.py)
#   - evaluation/reports/ragas_report.json          (ragas/eval_ragas.py)
#   - evaluation/reports/load_test_report.csv       (load_testing/load_test.py)
#
# No third-party dependencies: charts are inline SVG strings.
#
# Usage:
#   python -m evaluation.dashboard.dashboard_generator
#   python -m evaluation.dashboard.dashboard_generator --output-dir evaluation/dashboard

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "evaluation" / "outputs"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"
DEFAULT_OUT_HTML = Path(__file__).resolve().parent / "index.html"

# Metrics worth a headline bar chart.
_KEY_METRICS = [
    ("mrr", "MRR"),
    ("recall_at_5", "Recall@5"),
    ("ndcg_at_5", "NDCG@5"),
    ("precision_at_5", "Precision@5"),
    ("hit_at_5", "Hit@5"),
    ("routing_accuracy", "Routing accuracy"),
    ("citation_coverage", "Citation coverage"),
]

_RAGAS_METRICS = [
    ("Faithfulness", "Faithfulness"),
    ("Answer Relevancy", "Answer Relevancy"),
    ("Context Precision", "Context Precision"),
]

_LOAD_COLS = [
    ("users", "Users"),
    ("total_requests", "Requests"),
    ("errors", "Errors"),
    ("avg_ttft", "Avg TTFB (s)"),
    ("avg_latency", "Avg latency (s)"),
    ("p95_latency", "p95 latency (s)"),
    ("throughput_rps", "Throughput (rps)"),
]


def _esc(value: Any) -> str:
    """HTML-escape any value for safe embedding."""
    return html.escape(str(value))


def _fmt(value: Any) -> str:
    """Format a numeric value for display."""
    if isinstance(value, (int, float)):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return _esc(value)


# ---------------------------------------------------------------------------
# Loading (reads existing outputs only)
# ---------------------------------------------------------------------------

def load_benchmark_reports(outputs_dir: Path) -> List[Dict[str, Any]]:
    """Load all evaluation_report_*.json files, oldest first."""
    reports = []
    if not outputs_dir.is_dir():
        return reports
    for path in sorted(outputs_dir.glob("evaluation_report_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            reports.append(
                {
                    "path": path.name,
                    "timestamp": data.get("timestamp", path.name),
                    "total_samples": data.get("total_samples", 0),
                    "metrics": data.get("aggregated_metrics", {}),
                }
            )
        except (json.JSONDecodeError, OSError) as exc:
            reports.append({"path": path.name, "error": str(exc), "metrics": {}})
    return reports


def load_ragas(reports_dir: Path) -> Dict[str, Any]:
    """Load ragas_report.json if present."""
    path = reports_dir / "ragas_report.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_load_test(reports_dir: Path) -> List[Dict[str, Any]]:
    """Load load_test_report.csv rows if present."""
    path = reports_dir / "load_test_report.csv"
    if not path.exists():
        return []
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                parsed = {}
                for k, v in row.items():
                    try:
                        parsed[k] = float(v)
                    except (TypeError, ValueError):
                        parsed[k] = v
                rows.append(parsed)
    except OSError:
        return []
    return rows


# ---------------------------------------------------------------------------
# Charts (inline SVG, dependency-free)
# ---------------------------------------------------------------------------

def svg_bars(items: List[Tuple[str, float]], width: int = 420, height: int = 160) -> str:
    """Render a horizontal bar chart as an SVG string."""
    if not items:
        return '<p class="no-data">No data</p>'
    usable = [i for i in items if isinstance(i[1], (int, float))]
    if not usable:
        return '<p class="no-data">No numeric data</p>'
    max_val = max(abs(v) for _, v in usable) or 1.0
    label_w = 150
    bar_w = width - label_w - 60
    row_h = max(22, height // max(len(usable), 1))
    chart_h = row_h * len(usable) + 12
    parts = [f'<svg viewBox="0 0 {width} {chart_h}" class="chart" role="img">']
    for i, (label, value) in enumerate(usable):
        y = 8 + i * row_h
        bar_len = max(2, int(abs(value) / max_val * bar_w))
        parts.append(
            f'<text x="2" y="{y + 12}" class="bar-label">{_esc(label)[:24]}</text>'
            f'<rect x="{label_w}" y="{y}" width="{bar_len}" height="14" class="bar"/>'
            f'<text x="{label_w + bar_len + 6}" y="{y + 12}" class="bar-val">{_fmt(value)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _metric_value(value: Any) -> Any:
    """Return the display value for a metric.

    Newer reports store {mean, p95, min, max, n} per metric; legacy reports
    store a flat float.  Both are supported for backward compatibility.
    """
    if isinstance(value, dict):
        return value.get("mean", value.get("p95", 0.0))
    return value


def _metric_bars(metrics: Dict[str, Any], pairs: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
    """Extract (label, value) pairs present in the metrics dict."""
    items = []
    for key, label in pairs:
        value = _metric_value(metrics.get(key))
        if isinstance(value, (int, float)):
            items.append((label, float(value)))
    return items


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def _section(title: str, body: str) -> str:
    return f'<section class="card"><h2>{_esc(title)}</h2>{body}</section>'


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    if not rows:
        return '<p class="no-data">No data</p>'
    thead = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    trs = []
    for row in rows:
        tds = "".join(f"<td>{_esc(c)}</td>" for c in row)
        trs.append(f"<tr>{tds}</tr>")
    return f'<table><thead><tr>{thead}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def _build_html(
    benchmark_reports: List[Dict[str, Any]],
    ragas: Dict[str, Any],
    load_test_rows: List[Dict[str, Any]],
) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Latest benchmark report (last in the sorted list).
    latest = benchmark_reports[-1] if benchmark_reports else None
    latest_metrics = (latest or {}).get("metrics", {})

    sections = [
        f"<h1>LectureMIND Benchmark Dashboard</h1>",
        f'<p class="meta">Generated {_esc(generated)} · reads existing evaluation outputs only — '
        f"no evaluations were re-run.</p>",
    ]

    # 1. Latest benchmark metrics.
    if latest:
        body = f'<p class="meta">Report: <code>{_esc(latest["path"])}</code> · '
        body += f'samples: {_esc(latest.get("total_samples", 0))}</p>'
        bars = _metric_bars(latest_metrics, _KEY_METRICS)
        body += svg_bars(bars)
        if latest_metrics:
            body += _table(
                ["Metric", "Value"],
                [[k, _fmt(_metric_value(v))] for k, v in sorted(latest_metrics.items())],
            )
        sections.append(_section("Latest benchmark (aggregated)", body))
    else:
        sections.append(_section("Latest benchmark (aggregated)", '<p class="no-data">No benchmark reports found in evaluation/outputs/.</p>'))

    # 2. RAGAS.
    if ragas:
        items = []
        for key, label in _RAGAS_METRICS:
            value = ragas.get(key)
            if isinstance(value, (int, float)):
                items.append((label, float(value)))
        body = svg_bars(items)
        body += _table(["Metric", "Value"], [[k, _fmt(v)] for k, v in ragas.items()])
        sections.append(_section("RAGAS answer quality", body))
    else:
        sections.append(_section("RAGAS answer quality", '<p class="no-data">No ragas_report.json found in evaluation/reports/.</p>'))

    # 3. Load test.
    if load_test_rows:
        body = svg_bars(
            [(f"{int(r.get('users', 0))} users · p95", float(r.get("p95_latency", 0) or 0)) for r in load_test_rows]
        )
        body += _table(
            [h for _, h in _LOAD_COLS],
            [[row.get(k, "") for k, _ in _LOAD_COLS] for row in load_test_rows],
        )
        sections.append(_section("Load test (100/500/1000 users)", body))
    else:
        sections.append(_section("Load test (100/500/1000 users)", '<p class="no-data">No load_test_report.csv found in evaluation/reports/.</p>'))

    # 4. History.
    if len(benchmark_reports) > 1:
        rows = []
        for rep in benchmark_reports:
            m = rep.get("metrics", {})
            rows.append(
                [
                    rep.get("timestamp", ""),
                    rep.get("total_samples", ""),
                    _fmt(_metric_value(m.get("mrr", "—"))),
                    _fmt(_metric_value(m.get("recall_at_5", "—"))),
                    _fmt(_metric_value(m.get("ndcg_at_5", "—"))),
                    _fmt(_metric_value(m.get("routing_accuracy", "—"))),
                ]
            )
        sections.append(
            _section("Run history", _table(["Timestamp", "Samples", "MRR", "Recall@5", "NDCG@5", "Routing"], rows))
        )

    css = """
    body { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
           margin: 0; background: #f5f6f8; color: #1a202c; }
    header, main { max-width: 960px; margin: 0 auto; padding: 16px 20px; }
    h1 { font-size: 22px; margin: 20px 0 4px; }
    .meta { color: #64748b; font-size: 13px; }
    .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
            padding: 16px 20px; margin: 16px 0; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
    .card h2 { font-size: 15px; margin: 0 0 12px; color: #334155; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 10px; }
    th, td { border: 1px solid #e2e8f0; padding: 6px 10px; text-align: left; }
    th { background: #f8fafc; font-weight: 600; }
    .bar-label { font-size: 11px; fill: #475569; }
    .bar-val { font-size: 11px; fill: #334155; }
    .bar { fill: #4f46e5; rx: 3; }
    .no-data { color: #94a3b8; font-style: italic; font-size: 13px; }
    code { background: #f1f5f9; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
    """

    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>LectureMIND Benchmark Dashboard</title><style>{css}</style></head>"
        f"<body><header>{sections[0]}{sections[1]}</header><main>"
        + "".join(sections[2:])
        + "</main></body></html>"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_dashboard(
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    out_html: Path = DEFAULT_OUT_HTML,
) -> Path:
    """Generate the dashboard HTML from existing outputs. Returns the path."""
    benchmark_reports = load_benchmark_reports(outputs_dir)
    ragas = load_ragas(reports_dir)
    load_test_rows = load_load_test(reports_dir)

    html_doc = _build_html(benchmark_reports, ragas, load_test_rows)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_doc, encoding="utf-8")
    return out_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the benchmark dashboard from existing outputs.")
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_HTML)
    args = parser.parse_args()
    path = generate_dashboard(args.outputs_dir, args.reports_dir, args.out)
    print(f"Dashboard written to {path}")


if __name__ == "__main__":
    main()
