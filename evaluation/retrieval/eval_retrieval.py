# evaluation/retrieval/eval_retrieval.py
# Retrieval evaluation using ranx.
#
# Measures MRR, Recall@5, Recall@10, nDCG@10.
# Runs with reranker and without reranker.
# Generates comparison report in evaluation/reports/.
#
# Completely independent from frontend.

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> List[Dict]:
    """Load evaluation dataset (JSONL format: {query, relevant_chunk_ids})."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def split_dataset(
    data: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple:
    """Split dataset into train/val/test (70/15/15)."""
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]


def evaluate_retrieval(
    queries: List[Dict],
    retrieve_fn: Any,
    k_values: List[int] = [5, 10],
    ranx_module: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Evaluate retrieval quality.

    Args:
        queries: List of {query, relevant_chunk_ids}.
        retrieve_fn: callable(query) -> [chunk_id, ...].
        k_values: K values for Recall@K and nDCG@K.
        ranx_module: Optional ranx module (for testing).

    Returns:
        Dict of metric_name -> score.
    """
    if ranx_module is None:
        try:
            import ranx
            ranx_module = ranx
        except ImportError:
            logger.warning("ranx not installed. Using fallback metrics.")
            return _fallback_metrics(queries, retrieve_fn, k_values)

    from ranx import Qrels, Run, evaluate

    qrels_dict = {}
    run_dict = {}

    for i, item in enumerate(queries):
        qid = f"q_{i}"
        qrels_dict[qid] = {cid: 1 for cid in item["relevant_chunk_ids"]}
        retrieved = retrieve_fn(item["query"])
        run_dict[qid] = {cid: float(len(retrieved) - rank) for rank, cid in enumerate(retrieved)}

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    metrics = {}
    metrics["MRR"] = evaluate(qrels, run, "mrr")

    for k in k_values:
        metrics[f"Recall@{k}"] = evaluate(qrels, run, f"recall@{k}")
    metrics["nDCG@10"] = evaluate(qrels, run, "ndcg@10")

    return metrics


def _fallback_metrics(queries, retrieve_fn, k_values):
    """Fallback when ranx is not installed."""
    total_mrr = 0
    recall_at = {k: 0 for k in k_values}

    for item in queries:
        relevant = set(item["relevant_chunk_ids"])
        retrieved = retrieve_fn(item["query"])

        # MRR
        for rank, cid in enumerate(retrieved, 1):
            if cid in relevant:
                total_mrr += 1.0 / rank
                break

        # Recall@K
        for k in k_values:
            hits = len(relevant & set(retrieved[:k]))
            recall_at[k] += hits / max(len(relevant), 1)

    n = max(len(queries), 1)
    metrics = {"MRR": total_mrr / n}
    for k in k_values:
        metrics[f"Recall@{k}"] = recall_at[k] / n
    metrics["nDCG@10"] = 0.0  # Simplified fallback.
    return metrics


def run_comparison(
    test_queries: List[Dict],
    retrieve_with_reranker: Any,
    retrieve_without_reranker: Any,
    report_dir: Path = Path("evaluation/reports"),
) -> Dict:
    """Run with/without reranker comparison and save report."""
    report_dir.mkdir(parents=True, exist_ok=True)

    results_with = evaluate_retrieval(test_queries, retrieve_with_reranker)
    results_without = evaluate_retrieval(test_queries, retrieve_without_reranker)

    report = {
        "with_reranker": results_with,
        "without_reranker": results_without,
    }

    report_path = report_dir / "retrieval_comparison.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Retrieval report saved to %s", report_path)

    return report
