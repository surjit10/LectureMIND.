# evaluation/metrics/latency_metrics.py
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def extract_latencies(telemetry: Dict[str, float]) -> Dict[str, float]:
    """
    Extracts latencies from a telemetry object.
    Expects timestamps or durations for planner, retrieval, reranker, and generation.
    """
    if not telemetry:
        return {
            "planner_latency": 0.0,
            "retrieval_latency": 0.0,
            "reranker_latency": 0.0,
            "generation_latency": 0.0,
            "total_latency": 0.0
        }
        
    return {
        "planner_latency": telemetry.get("planner_latency", 0.0),
        "retrieval_latency": telemetry.get("retrieval_latency", 0.0),
        "reranker_latency": telemetry.get("reranker_latency", 0.0),
        "generation_latency": telemetry.get("generation_latency", 0.0),
        "total_latency": telemetry.get("total_latency", 0.0),
    }
