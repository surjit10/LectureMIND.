# evaluation/reports/report_generator.py
import json
import csv
import logging
import statistics
from typing import Any, Dict, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def _p95(values: List[float]) -> float:
    """Return the 95th percentile of a numeric list (nearest-rank)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(0.95 * len(sorted_vals)) - 1)
    return sorted_vals[idx]


class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate(self, results: List[Dict[str, Any]]) -> None:
        """Generates evaluation reports in JSON, CSV, and Markdown formats."""
        if not results:
            logger.warning("No results to generate reports for.")
            return

        aggregated_metrics = self._aggregate_metrics(results)
        breakdowns = self._breakdowns(results)

        self._generate_json(results, aggregated_metrics, breakdowns)
        self._generate_csv(results)
        self._generate_markdown(aggregated_metrics, breakdowns, results)

        logger.info(f"Reports generated successfully in {self.output_dir}")

    def _numeric_metric_keys(self, results: List[Dict[str, Any]]) -> List[str]:
        """Union of all numeric metric keys across results, in stable order."""
        keys: List[str] = []
        seen = set()
        for res in results:
            for key, value in res.get("metrics", {}).items():
                if isinstance(value, (int, float)) and key not in seen:
                    seen.add(key)
                    keys.append(key)
        return keys

    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mean, p95, min, max per numeric metric across all results."""
        keys = self._numeric_metric_keys(results)
        aggregated: Dict[str, Any] = {}
        for key in keys:
            values = []
            for res in results:
                v = res.get("metrics", {}).get(key)
                if isinstance(v, (int, float)):
                    values.append(float(v))
            if not values:
                continue
            aggregated[key] = {
                "mean": sum(values) / len(values),
                "p95": _p95(values),
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
        return aggregated

    def _breakdowns(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Mean of each numeric metric grouped by question_type and difficulty."""
        keys = self._numeric_metric_keys(results)
        by_type: Dict[str, Dict[str, List[float]]] = {}
        by_difficulty: Dict[str, Dict[str, List[float]]] = {}

        for res in results:
            qtype = res.get("question_type", "unknown")
            diff = res.get("difficulty", "unknown")
            for key in keys:
                v = res.get("metrics", {}).get(key)
                if not isinstance(v, (int, float)):
                    continue
                by_type.setdefault(qtype, {}).setdefault(key, []).append(float(v))
                by_difficulty.setdefault(diff, {}).setdefault(key, []).append(float(v))

        def _meanize(groups: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
            out = {}
            for group, metrics in groups.items():
                out[group] = {k: sum(v) / len(v) for k, v in metrics.items()}
            return out

        return {
            "by_question_type": _meanize(by_type),
            "by_difficulty": _meanize(by_difficulty),
        }

    def _generate_json(
        self,
        results: List[Dict[str, Any]],
        aggregated_metrics: Dict[str, Any],
        breakdowns: Dict[str, Any],
    ) -> None:
        output_path = self.output_dir / f"evaluation_report_{self.timestamp}.json"
        report_data = {
            "timestamp": self.timestamp,
            "total_samples": len(results),
            "successful_samples": sum(1 for r in results if not r.get("error")),
            "failed_samples": sum(1 for r in results if r.get("error")),
            "aggregated_metrics": aggregated_metrics,
            "breakdowns": breakdowns,
            "results": results
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"JSON report saved to {output_path}")

    def _generate_csv(self, results: List[Dict[str, Any]]) -> None:
        output_path = self.output_dir / f"evaluation_report_{self.timestamp}.csv"
        if not results:
            return

        csv_rows = []
        all_metric_keys = set()
        for res in results:
            row = {
                "lecture_id": res.get("lecture_id", ""),
                "query": res.get("query", ""),
                "question_type": res.get("question_type", ""),
                "difficulty": res.get("difficulty", ""),
                "expected_route": res.get("expected_route", ""),
                "error": res.get("error", ""),
            }
            metrics = res.get("metrics", {})
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    row[f"metric_{k}"] = v
                    all_metric_keys.add(f"metric_{k}")
            csv_rows.append(row)

        fieldnames = ["lecture_id", "query", "question_type", "difficulty", "expected_route", "error"] + sorted(list(all_metric_keys))

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info(f"CSV report saved to {output_path}")

    def _generate_markdown(
        self,
        aggregated_metrics: Dict[str, Any],
        breakdowns: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> None:
        output_path = self.output_dir / f"evaluation_report_{self.timestamp}.md"

        md_content = [
            "# LectureMind Evaluation Report",
            f"**Generated:** {self.timestamp}",
            f"**Samples:** {len(results)} ({sum(1 for r in results if not r.get('error'))} successful, "
            f"{sum(1 for r in results if r.get('error'))} failed)\n",
            "## Aggregated Metrics\n",
            "| Metric | Mean | p95 | Min | Max | n |",
            "|---|---|---|---|---|---|",
        ]

        for key in sorted(aggregated_metrics.keys()):
            agg = aggregated_metrics[key]
            md_content.append(
                f"| {key} | {agg['mean']:.4f} | {agg['p95']:.4f} | {agg['min']:.4f} | {agg['max']:.4f} | {agg['n']} |"
            )

        # Breakdowns
        for section, groups in breakdowns.items():
            if not groups:
                continue
            md_content.append(f"\n## {section.replace('_', ' ').title()}\n")
            md_content.append("| Group | Metric | Mean |")
            md_content.append("|---|---|---|")
            for group, metrics in sorted(groups.items()):
                for key in sorted(metrics.keys()):
                    md_content.append(f"| {group} | {key} | {metrics[key]:.4f} |")

        # Per-sample table (key metrics only)
        md_content.append("\n## Per-Sample Results\n")
        md_content.append("| # | Type | Difficulty | Query | Route | MRR | Recall@5 | Answer F1 | Citation | Total Lat (s) |")
        md_content.append("|---|---|---|---|---|---|---|---|---|---|")
        for i, res in enumerate(results, 1):
            m = res.get("metrics", {})
            query = (res.get("query", "") or "")[:50].replace("|", "/")
            md_content.append(
                f"| {i} | {res.get('question_type','')} | {res.get('difficulty','')} | {query} | "
                f"{res.get('expected_route','')} | {m.get('mrr', 0.0):.3f} | {m.get('recall_at_5', 0.0):.3f} | "
                f"{m.get('answer_f1', 0.0):.3f} | {m.get('citation_coverage', 0.0):.3f} | {m.get('total_latency', 0.0):.2f} |"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content) + "\n")
        logger.info(f"Markdown report saved to {output_path}")
