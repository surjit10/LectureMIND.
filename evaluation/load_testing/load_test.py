# evaluation/load_testing/load_test.py
# Load testing — measures TTFT, Latency, Throughput, GPU Utilization.
#
# Tests: 100, 500, 1000 concurrent users.
# Generates CSV report.

import csv
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

USER_LEVELS = [100, 500, 1000]


def _send_query(api_url: str, query: str) -> Dict:
    """Send a single query and measure latency."""
    import urllib.request

    start = time.perf_counter()
    data = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        f"{api_url}/query",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            first_byte = time.perf_counter()
            body = resp.read()
            end = time.perf_counter()
            return {
                "ttft": first_byte - start,
                "latency": end - start,
                "status": resp.status,
                "error": None,
            }
    except Exception as exc:
        end = time.perf_counter()
        return {
            "ttft": end - start,
            "latency": end - start,
            "status": 0,
            "error": str(exc),
        }


def run_load_test(
    api_url: str = "http://localhost:8000",
    queries: Optional[List[str]] = None,
    user_levels: List[int] = USER_LEVELS,
    report_dir: Path = Path("evaluation/reports"),
) -> List[Dict]:
    """
    Run load tests at 100, 500, 1000 concurrent users.

    Returns:
        List of result dicts per user level.
    """
    if queries is None:
        queries = [
            "Why BFS before Dijkstra?",
            "Summarize graph algorithms",
            "What is prerequisite of DP?",
            "Explain recursion",
            "How does merge sort work?",
        ]

    report_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for level in user_levels:
        logger.info("Load test: %d concurrent users...", level)
        test_queries = [queries[i % len(queries)] for i in range(level)]

        start = time.perf_counter()
        results = []

        with ThreadPoolExecutor(max_workers=min(level, 50)) as executor:
            futures = {executor.submit(_send_query, api_url, q): q for q in test_queries}
            for future in as_completed(futures):
                results.append(future.result())

        elapsed = time.perf_counter() - start
        latencies = [r["latency"] for r in results if r["error"] is None]
        ttfts = [r["ttft"] for r in results if r["error"] is None]
        errors = sum(1 for r in results if r["error"] is not None)

        summary = {
            "users": level,
            "total_requests": level,
            "successful": level - errors,
            "errors": errors,
            "avg_ttft": sum(ttfts) / max(len(ttfts), 1),
            "avg_latency": sum(latencies) / max(len(latencies), 1),
            "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "throughput_rps": (level - errors) / max(elapsed, 0.01),
            "gpu_utilization": "N/A",  # Requires nvidia-smi integration.
        }
        all_results.append(summary)
        logger.info("  → %d users: avg_latency=%.2fs, throughput=%.1f rps", level, summary["avg_latency"], summary["throughput_rps"])

    # Write CSV report.
    csv_path = report_dir / "load_test_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    logger.info("Load test report saved to %s", csv_path)
    return all_results
