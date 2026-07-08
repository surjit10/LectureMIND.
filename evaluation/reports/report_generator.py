# evaluation/reports/report_generator.py
import json
import csv
import logging
from typing import Any, Dict, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

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

        self._generate_json(results, aggregated_metrics)
        self._generate_csv(results)
        self._generate_markdown(aggregated_metrics)
        
        logger.info(f"Reports generated successfully in {self.output_dir}")

    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculates mean for all numeric metrics across all results."""
        aggregated = {}
        counts = {}
        for res in results:
            metrics = res.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    aggregated[key] = aggregated.get(key, 0.0) + value
                    counts[key] = counts.get(key, 0) + 1
                    
        return {k: v / counts[k] for k, v in aggregated.items() if counts[k] > 0}

    def _generate_json(self, results: List[Dict[str, Any]], aggregated_metrics: Dict[str, float]) -> None:
        output_path = self.output_dir / f"evaluation_report_{self.timestamp}.json"
        report_data = {
            "timestamp": self.timestamp,
            "total_samples": len(results),
            "aggregated_metrics": aggregated_metrics,
            "results": results
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"JSON report saved to {output_path}")

    def _generate_csv(self, results: List[Dict[str, Any]]) -> None:
        output_path = self.output_dir / f"evaluation_report_{self.timestamp}.csv"
        if not results:
            return
            
        # Flatten the results dict for CSV format
        csv_rows = []
        all_metric_keys = set()
        for res in results:
            row = {
                "lecture_id": res.get("lecture_id", ""),
                "query": res.get("query", ""),
                "expected_route": res.get("expected_route", ""),
                "error": res.get("error", "")
            }
            metrics = res.get("metrics", {})
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    row[f"metric_{k}"] = v
                    all_metric_keys.add(f"metric_{k}")
            csv_rows.append(row)
            
        fieldnames = ["lecture_id", "query", "expected_route", "error"] + sorted(list(all_metric_keys))
        
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info(f"CSV report saved to {output_path}")

    def _generate_markdown(self, aggregated_metrics: Dict[str, float]) -> None:
        output_path = self.output_dir / f"evaluation_report_{self.timestamp}.md"
        
        md_content = [
            "# LectureMind Evaluation Report",
            f"**Generated:** {self.timestamp}\n",
            "## Aggregated Metrics\n",
            "| Metric | Value |",
            "|---|---|"
        ]
        
        for key, value in sorted(aggregated_metrics.items()):
            # Format float nicely
            formatted_val = f"{value:.4f}" if isinstance(value, float) else str(value)
            md_content.append(f"| {key} | {formatted_val} |")
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content) + "\n")
        logger.info(f"Markdown report saved to {output_path}")
