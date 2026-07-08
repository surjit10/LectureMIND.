# evaluation/metrics/planner_metrics.py
import logging

logger = logging.getLogger(__name__)

def calculate_routing_accuracy(expected_route: str, actual_route: str) -> float:
    """Calculates if the planner selected the expected retrieval route."""
    if expected_route is None or actual_route is None:
        return 0.0
    return 1.0 if expected_route.lower() == actual_route.lower() else 0.0
