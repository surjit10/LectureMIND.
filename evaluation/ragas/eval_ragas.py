# evaluation/ragas/eval_ragas.py
# RAGAS evaluation — Faithfulness, Answer Relevancy, Context Precision.
#
# Completely independent from frontend.

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def evaluate_ragas(
    test_data: List[Dict[str, str]],
    query_fn: Any,
    report_dir: Path = Path("evaluation/reports"),
    ragas_module: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Run RAGAS evaluation.

    Args:
        test_data: List of {query, ground_truth, context}.
        query_fn: callable(query) -> {answer, context, ...}.
        report_dir: Directory for reports.
        ragas_module: Optional ragas module for testing.

    Returns:
        Dict of metric_name -> score.

    Metrics:
        Faithfulness, Answer Relevancy, Context Precision
    """
    if ragas_module is None:
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision
            from datasets import Dataset
        except ImportError:
            logger.warning("ragas/datasets not installed. Using placeholder scores.")
            return _placeholder_scores(test_data)

    # Build evaluation dataset.
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in test_data:
        result = query_fn(item["query"])
        questions.append(item["query"])
        answers.append(result.get("answer", ""))
        contexts.append([result.get("context", "")])
        ground_truths.append([item.get("ground_truth", "")])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    metrics = {
        "Faithfulness": results["faithfulness"],
        "Answer Relevancy": results["answer_relevancy"],
        "Context Precision": results["context_precision"],
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "ragas_report.json"
    report_path.write_text(json.dumps(metrics, indent=2))
    logger.info("RAGAS report saved to %s", report_path)

    return metrics


def _placeholder_scores(test_data: List[Dict]) -> Dict[str, float]:
    """Placeholder scores when ragas is not installed."""
    return {
        "Faithfulness": 0.0,
        "Answer Relevancy": 0.0,
        "Context Precision": 0.0,
    }


def main() -> int:
    """CLI entry point: run RAGAS over a live LectureMIND server.

    Usage:
        python -m evaluation.ragas.eval_ragas [--api-url http://localhost:8000]

    Requires:
        - A running server (POST /query) with an active lecture.
        - The optional dependencies ``ragas`` and ``datasets``
          (pip install ragas datasets).

    Writes evaluation/reports/ragas_report.json on success.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="RAGAS answer-quality evaluation")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    try:
        import ragas  # noqa: F401
        import datasets  # noqa: F401
    except ImportError:
        logger.error(
            "ragas/datasets are not installed. Install them with: "
            "pip install ragas datasets. Refusing to write a report of "
            "placeholder zeros."
        )
        return 2

    def _query(query: str) -> Dict[str, str]:
        import urllib.request

        data = json.dumps({"query": query}).encode()
        req = urllib.request.Request(
            f"{args.api_url}/query",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode())
        return {
            "answer": payload.get("answer", ""),
            "context": "\n\n".join(
                str(src.get("chunk_id", "")) for src in payload.get("sources", [])
            ),
        }

    test_data = [
        {"query": "What is the main topic of the lecture?", "ground_truth": ""},
    ]
    metrics = evaluate_ragas(test_data, _query)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
