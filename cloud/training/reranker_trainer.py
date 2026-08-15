# cloud/training/reranker_trainer.py
# Stage B2 — Reranker Fine-tuning.
#
# Fine-tunes BAAI/bge-reranker-large as a CrossEncoder using
# MultipleNegativesRankingLoss on triplets from B1.
#
# Input: cloud_runtime/lectures/{lecture_id}/triplets.json
# Output: cloud_runtime/lectures/{lecture_id}/reranker_model/
#         cloud_runtime/lectures/{lecture_id}/training_metrics.json
# Environment: Kaggle GPU only.

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import CloudSettings
from schemas.triplet import RerankerTriplet

logger = logging.getLogger(__name__)

RERANKER_BASE_MODEL = "BAAI/bge-reranker-base"


def _load_triplets(lecture_dir: Path) -> List[Dict[str, Any]]:
    """Load and validate triplets from triplets.json."""
    triplets_path = lecture_dir / "triplets.json"
    if not triplets_path.exists():
        raise FileNotFoundError(f"B2: triplets.json not found: {triplets_path}")

    triplets_data = json.loads(triplets_path.read_text(encoding="utf-8"))

    # Validate each triplet against Chunk 1 schema.
    for t in triplets_data:
        RerankerTriplet(**t)

    if not triplets_data:
        raise ValueError("B2: triplets.json contains zero triplets.")

    return triplets_data


def train_reranker(
    lecture_id: str,
    cloud_settings: CloudSettings | None = None,
    trainer_fn: Optional[Any] = None,
    base_model: str = RERANKER_BASE_MODEL,
    epochs: int = 3,
    batch_size: int = 8,
    warmup_ratio: float = 0.1,
    learning_rate: Optional[float] = None,
    evaluation_steps: int = 0,
    dev_triplets: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Fine-tune the reranker cross-encoder on generated triplets.

    Args:
        lecture_id: Unique lecture identifier.
        cloud_settings: Injected CloudSettings.
        trainer_fn: Optional callable that performs the actual training.
                    Signature: trainer_fn(triplets, output_dir, base_model, **kwargs)
                    Returns: metrics dict. Used for test injection.
        base_model: HuggingFace model name for the base reranker.
        epochs: Number of training epochs.
        batch_size: Training batch size (small for T4 VRAM).
        warmup_ratio: Learning rate warmup ratio.
        learning_rate: Optional AdamW learning rate. None → sentence-transformers
            default (2e-5). Additive; existing callers are unaffected.
        evaluation_steps: When > 0 AND dev_triplets provided, run the
            CERerankingEvaluator on dev triplets every N training steps
            (and trigger best-model saving). 0 disables mid-training eval.
        dev_triplets: Optional held-out triplets for mid-training evaluation
            and loss tracking. Not used by the injected trainer_fn path.

    Returns:
        Training metrics dict with MRR, Recall@K, nDCG@10.
    """
    settings = cloud_settings or CloudSettings()
    lecture_dir = Path(settings.lecture_dir(lecture_id))

    # Load triplets.
    triplets_data = _load_triplets(lecture_dir)
    logger.info("B2: Loaded %d triplets for training.", len(triplets_data))

    # Output directory.
    model_output_dir = lecture_dir / "reranker_model"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Train — either via injected function or real training.
    if trainer_fn is not None:
        metrics = trainer_fn(
            triplets_data, model_output_dir, base_model,
            epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, evaluation_steps=evaluation_steps,
        )
    else:
        metrics = _run_training(
            triplets_data, model_output_dir, base_model,
            epochs=epochs, batch_size=batch_size,
            warmup_ratio=warmup_ratio,
            learning_rate=learning_rate,
            evaluation_steps=evaluation_steps,
            dev_triplets=dev_triplets,
        )

    # Write training_metrics.json — simple JSON, no new schema.
    metrics_path = lecture_dir / "training_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    logger.info("B2: training_metrics.json written to %s", metrics_path)
    logger.info("B2: reranker_model/ saved to %s", model_output_dir)

    return metrics


def _run_training(
    triplets: List[Dict[str, Any]],
    output_dir: Path,
    base_model: str,
    epochs: int = 3,
    batch_size: int = 8,
    warmup_ratio: float = 0.1,
    learning_rate: Optional[float] = None,
    evaluation_steps: int = 0,
    dev_triplets: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """
    Execute real reranker fine-tuning using sentence-transformers CrossEncoder.

    Uses MultipleNegativesRankingLoss with small batch sizes and
    gradient accumulation for Kaggle T4 compatibility.
    """
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader
    import math

    # Build training examples.
    train_examples = []
    for t in triplets:
        # Positive pair: (query, positive) → label 1
        train_examples.append(InputExample(texts=[t["query"], t["positive"]], label=1.0))
        # Negative pair: (query, negative) → label 0
        train_examples.append(InputExample(texts=[t["query"], t["negative"]], label=0.0))

    # Load CrossEncoder.
    model = CrossEncoder(base_model, num_labels=1)

    # Training data loader.
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
    )

    # Warmup steps.
    warmup_steps = math.ceil(len(train_dataloader) * epochs * warmup_ratio)

    # Train.
    logger.info(
        "B2: Training CrossEncoder — %d examples, %d epochs, batch=%d, warmup=%d steps",
        len(train_examples), epochs, batch_size, warmup_steps,
    )

    # Optional tuning knobs — additive; defaults reproduce legacy behavior.
    fit_kwargs = dict(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    if learning_rate is not None:
        fit_kwargs["optimizer_params"] = {"lr": learning_rate}
    if dev_triplets and evaluation_steps and evaluation_steps > 0:
        from sentence_transformers.cross_encoder.evaluation import (
            CERerankingEvaluator,
        )
        samples = [
            {
                "query": t["query"],
                "positive": [t["positive"]],
                "negative": [t["negative"]],
            }
            for t in dev_triplets
        ]
        evaluator = CERerankingEvaluator(samples, name="dev", mrr_at_k=10, ndcg_at_k=10)
        fit_kwargs["evaluator"] = evaluator
        fit_kwargs["evaluation_steps"] = evaluation_steps
        logger.info(
            "B2: Mid-training evaluation on %d dev triplets every %d steps.",
            len(dev_triplets), evaluation_steps,
        )

    fit_result = model.fit(**fit_kwargs)

    # Explicitly persist final model artifacts.
    # Safe because it operates on the already-trained in-memory model.
    # Does not alter optimizer state, training logic, metrics,
    # retrieval behavior, or package format.
    model.save(str(output_dir))

    # Evaluate — compute metrics on the training data as a baseline.
    # In production, a held-out eval set would be used.
    metrics = compute_reranker_metrics(model, triplets)

    # Loss history from CrossEncoder.fit's return value (list of per-step
    # loss dicts on recent sentence-transformers). Defensive: older versions
    # return nothing, in which case the key is simply omitted.
    loss_history = _extract_loss_history(fit_result)
    if loss_history:
        metrics["train_loss_history"] = loss_history

    return metrics


def _extract_loss_history(fit_result: Any) -> List[float]:
    """Extract a flat loss-history list from CrossEncoder.fit's return value.

    Recent sentence-transformers return a list of {"loss": float, ...} dicts;
    other versions return None or a plain list of floats. Never raise.
    """
    if not isinstance(fit_result, (list, tuple)):
        return []
    losses = []
    for item in fit_result:
        if isinstance(item, dict):
            loss = item.get("loss")
            if isinstance(loss, (int, float)):
                losses.append(float(loss))
        elif isinstance(item, (int, float)):
            losses.append(float(item))
    return losses


def compute_reranker_metrics(
    model: Any,
    triplets: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute MRR, Recall@K, nDCG@10 on the given triplets.

    For each triplet, scores positive and negative against query,
    then computes ranking metrics.

    Public — also used by Pipeline B (cloud/reranker_training/pipeline.py) to
    evaluate the trained model on a held-out dev set.
    """
    import numpy as np

    reciprocal_ranks = []
    recall_at_5 = []
    recall_at_10 = []
    ndcg_at_10 = []

    for t in triplets:
        query = t["query"]
        pos = t["positive"]
        neg = t["negative"]

        # Score both candidates.
        scores = model.predict([(query, pos), (query, neg)])
        pos_score = float(scores[0])
        neg_score = float(scores[1])

        # Rank: positive should be ranked higher.
        if pos_score >= neg_score:
            rank = 1
        else:
            rank = 2

        reciprocal_ranks.append(1.0 / rank)
        recall_at_5.append(1.0 if rank <= 5 else 0.0)
        recall_at_10.append(1.0 if rank <= 10 else 0.0)

        # nDCG@10 for binary relevance with 2 items.
        dcg = 1.0 / np.log2(rank + 1)
        idcg = 1.0 / np.log2(2)  # ideal: positive at rank 1
        ndcg_at_10.append(float(dcg / idcg))

    metrics = {
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
        "recall_at_5": round(float(np.mean(recall_at_5)), 4),
        "recall_at_10": round(float(np.mean(recall_at_10)), 4),
        "ndcg_at_10": round(float(np.mean(ndcg_at_10)), 4),
        "num_triplets": len(triplets),
    }
    return metrics


# Backward-compatible alias — the legacy private name is retained so existing
# internal callers keep working unchanged.
_compute_metrics = compute_reranker_metrics
