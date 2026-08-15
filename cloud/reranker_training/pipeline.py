# cloud/reranker_training/pipeline.py
#
# Pipeline B — Global Reranker Training orchestrator (independent Kaggle workflow).
#
#   Knowledge Packages
#         ↓  (discover + extract ONLY triplets.json, ignore everything else)
#   Merge Dataset
#         ↓  (dedupe, per-lecture filter, deterministic train/dev split)
#   Train CrossEncoder        (reuses cloud/training/reranker_trainer.py verbatim)
#         ↓
#   Evaluate (dev MRR / Recall@K / NDCG via the trainer's metric function)
#         ↓
#   Versioned commit (reranker_models/v{N}/, best/, latest/)
#         ↓
#   Export global_reranker_v{N}.zip
#
# Never processes videos. The only input artifact from Pipeline A is the
# triplets.json file inside each knowledge package.
#
# Usage (Kaggle Notebook B):
#   python -m cloud.reranker_training.pipeline \
#       --packages-root /kaggle/input/my-packages \
#       --epochs 3 --batch-size 8 --learning-rate 2e-5 \
#       --eval-frequency 200 --dev-fraction 0.1

# ----------------------------------------------------------------------------
# TensorFlow / protobuf workaround (same as run_ingestion_pipeline.py)
# ----------------------------------------------------------------------------
# Kaggle ships TensorFlow, but this project pins protobuf==3.20.3 for
# PaddlePaddle 2.6.x compatibility. TF's generated protobuf code needs a newer
# protobuf (google.protobuf.runtime_version), so when transformers probes for
# TF via find_spec() and imports it (e.g. while loading sentence-transformers'
# CrossEncoder here), the process crashes with "cannot import name
# 'runtime_version' from 'google.protobuf'". Hiding TF via sys.modules makes
# find_spec("tensorflow") return None so transformers stays on PyTorch only.
# This subprocess does NOT inherit sys.modules from run_kaggle_reranker_training.py,
# hence the guard is repeated here.
import sys

sys.modules.setdefault("tensorflow", None)

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import CloudSettings

from cloud.reranker_training.dataset_builder import DatasetReport, build_dataset, train_dev_split
from cloud.reranker_training.export import export_version
from cloud.reranker_training.versioning import (
    commit_version,
    now_iso,
    next_version,
    update_pointers,
)

logger = logging.getLogger(__name__)

DEFAULT_PACKAGES_ROOT = "/kaggle/input"
DEFAULT_MODELS_ROOT = "/kaggle/working/reranker_models/"
DEFAULT_SCRATCH_ROOT = "/kaggle/working/reranker_training/scratch/"


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters + dataset knobs for one training run."""

    epochs: int = 3
    batch_size: int = 8
    learning_rate: Optional[float] = None   # None → sentence-transformers default (2e-5)
    warmup_ratio: float = 0.1
    evaluation_steps: int = 0               # 0 → no mid-training evaluation
    dev_fraction: float = 0.1
    seed: int = 42
    min_triplets: int = 0                   # per-package minimum
    max_per_lecture: Optional[int] = None   # per-package cap
    min_total_triplets: int = 10            # abort if the merged set is below this
    base_model: str = "BAAI/bge-reranker-base"
    primary_metric: str = "ndcg_at_10_dev"


@dataclass
class PipelineResult:
    """Outcome of one pipeline run (never raises for data problems)."""

    status: str = "aborted"                 # "completed" | "aborted"
    reason: str = ""
    version: Optional[int] = None
    models_root: Optional[str] = None
    model_zip: Optional[str] = None
    n_packages_found: int = 0
    n_packages_used: int = 0
    n_packages_skipped: int = 0
    skipped_sources: List[Dict[str, str]] = field(default_factory=list)
    lecture_ids: List[str] = field(default_factory=list)
    n_triplets: int = 0
    n_train_triplets: int = 0
    n_dev_triplets: int = 0
    train_metrics: Dict[str, Any] = field(default_factory=dict)
    dev_metrics: Dict[str, Any] = field(default_factory=dict)
    eval_error: Optional[str] = None
    metadata_path: Optional[str] = None


def _build_scratch_settings(scratch_root: Path) -> CloudSettings:
    """CloudSettings pointing the trainer's lecture_dir at the scratch root."""
    return CloudSettings(
        LECTURE_OUTPUT_DIR=str(Path(scratch_root) / "{lecture_id}/"),
        FRAME_DIR=str(Path(scratch_root) / "{lecture_id}/frames/"),
        LOG_DIR=str(Path(scratch_root) / "{lecture_id}/logs/"),
    )


def run_training_pipeline(
    packages_root: Path,
    models_root: Optional[Path] = None,
    scratch_root: Optional[Path] = None,
    config: Optional[TrainingConfig] = None,
    trainer_fn: Optional[Callable] = None,
    settings: Optional[CloudSettings] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> PipelineResult:
    """
    Execute the full Pipeline B run.

    Args:
        packages_root: Directory scanned for lecture knowledge packages.
        models_root: Versioned store destination (defaults to config default).
        scratch_root: Scratch workspace for the trainer (defaults to config default).
        config: TrainingConfig; defaults applied when None.
        trainer_fn: Optional injected trainer (test/CI only; never on Kaggle).
        settings: Optional pre-built CloudSettings (test/CI only).
        exclude_dirs: Directory names skipped during package discovery. None →
            the CloudSettings.RERANKER_TRAINING_EXCLUDE_DIRS default.

    Returns:
        PipelineResult — the run NEVER raises for missing packages, corrupt
        triplets, failed training, or failed evaluation; it logs and continues.
    """
    cfg = config or TrainingConfig()
    models_root = Path(models_root) if models_root is not None else Path(DEFAULT_MODELS_ROOT)
    scratch_root = Path(scratch_root) if scratch_root is not None else Path(DEFAULT_SCRATCH_ROOT)

    logger.info("=" * 60)
    logger.info("PIPELINE B — GLOBAL RERANKER TRAINING")
    logger.info("Packages root: %s", packages_root)
    logger.info("Models root:   %s", models_root)
    logger.info("Scratch root:  %s", scratch_root)
    logger.info("=" * 60)

    # ── Step 1: discover + merge dataset (reliability: skip & continue) ──
    if exclude_dirs is None:
        exclude_dirs = list(CloudSettings().RERANKER_TRAINING_EXCLUDE_DIRS)
    report: DatasetReport = build_dataset(
        Path(packages_root),
        exclude_dirs=exclude_dirs,
        min_triplets=cfg.min_triplets,
        max_per_lecture=cfg.max_per_lecture,
    )

    result = PipelineResult(
        n_packages_found=report.sources_found,
        n_packages_used=report.sources_used,
        n_packages_skipped=len(report.sources_skipped),
        skipped_sources=report.sources_skipped,
        n_triplets=report.n_triplets_after_dedupe,
        lecture_ids=report.lecture_ids,
        models_root=str(models_root),
    )

    if not report.triplets:
        result.reason = "No usable triplets found in any knowledge package."
        logger.error("Pipeline B: %s", result.reason)
        return result
    if len(report.triplets) < cfg.min_total_triplets:
        result.reason = (
            f"Only {len(report.triplets)} merged triplets (< min {cfg.min_total_triplets})."
        )
        logger.error("Pipeline B: %s", result.reason)
        return result

    # ── Step 2: deterministic train/dev split ──
    train_triplets, dev_triplets = train_dev_split(
        report.triplets, dev_fraction=cfg.dev_fraction, seed=cfg.seed,
    )
    result.n_train_triplets = len(train_triplets)
    result.n_dev_triplets = len(dev_triplets)
    logger.info(
        "Pipeline B: train=%d triplets, dev=%d triplets.",
        len(train_triplets), len(dev_triplets),
    )

    # ── Step 3: train (reuses cloud/training/reranker_trainer.py) ──
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    scratch_settings = settings or _build_scratch_settings(scratch_root)
    scratch_dir = Path(scratch_settings.lecture_dir(run_id))
    scratch_dir.mkdir(parents=True, exist_ok=True)

    (scratch_dir / "triplets.json").write_text(
        json.dumps(train_triplets, indent=2), encoding="utf-8"
    )
    if dev_triplets:
        (scratch_dir / "dev_triplets.json").write_text(
            json.dumps(dev_triplets, indent=2), encoding="utf-8"
        )

    from cloud.training.reranker_trainer import train_reranker

    try:
        train_metrics = train_reranker(
            run_id,
            cloud_settings=scratch_settings,
            trainer_fn=trainer_fn,
            base_model=cfg.base_model,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            warmup_ratio=cfg.warmup_ratio,
            learning_rate=cfg.learning_rate,
            evaluation_steps=cfg.evaluation_steps,
            dev_triplets=dev_triplets,
        )
        result.train_metrics = train_metrics
    except Exception as exc:
        # Interrupted/failed training must not crash the notebook — log and stop
        # cleanly WITHOUT committing any version (no half-written checkpoints).
        result.reason = f"Training failed: {exc}"
        logger.exception("Pipeline B: %s", result.reason)
        return result

    model_src = scratch_dir / "reranker_model"
    if not model_src.is_dir():
        result.reason = "Training finished but reranker_model/ was not produced."
        logger.error("Pipeline B: %s", result.reason)
        return result

    # ── Step 4: evaluate on held-out dev set (reuses the trainer's metrics) ──
    result.dev_metrics, result.eval_error = _evaluate_dev(model_src, dev_triplets)

    # ── Step 5: versioned commit + pointers ──
    version = next_version(models_root)
    metadata = _build_metadata(
        version=version,
        run_id=run_id,
        cfg=cfg,
        report=report,
        result=result,
    )
    evaluation_report = {
        "version": version,
        "train_metrics": result.train_metrics,
        "dev_metrics": result.dev_metrics,
        "eval_error": result.eval_error,
        "n_packages_used": report.sources_used,
        "n_triplets_total": len(report.triplets),
    }

    commit_version(models_root, version, model_src, metadata, evaluation_report)

    primary_value = _primary_value(result, cfg.primary_metric)
    update_pointers(
        models_root, version,
        primary_value=primary_value,
        primary_metric=cfg.primary_metric,
    )

    # ── Step 6: export the trained reranker only ──
    try:
        zip_path = export_version(models_root, version)
        result.model_zip = str(zip_path)
    except Exception as exc:
        logger.error(
            "Pipeline B: version committed but export failed (non-fatal): %s", exc,
        )

    result.status = "completed"
    result.version = version
    result.metadata_path = str(models_root / f"v{version}" / "metadata.json")
    logger.info(
        "Pipeline B: COMPLETED — version v%d, %d triplets, %s",
        version, len(report.triplets), result.model_zip,
    )
    return result


def _evaluate_dev(
    model_src: Path, dev_triplets: List[Dict[str, Any]],
) -> tuple:
    """Score the trained model against held-out triplets.

    Reuses cloud.training.reranker_trainer._compute_metrics (the same metric
    implementation used during training — no duplicated logic). A failed
    evaluation is logged and reported, never fatal.
    """
    if not dev_triplets:
        return {}, None
    try:
        from sentence_transformers import CrossEncoder

        from cloud.training.reranker_trainer import compute_reranker_metrics

        model = CrossEncoder(str(model_src))
        return compute_reranker_metrics(model, dev_triplets), None
    except Exception as exc:
        logger.warning("Pipeline B: dev evaluation failed (non-fatal): %s", exc)
        return {}, str(exc)


def _primary_value(result: PipelineResult, primary_metric: str) -> Optional[float]:
    """Resolve the comparison metric: prefer dev, fall back to train, else None."""
    dev_ndcg = result.dev_metrics.get("ndcg_at_10")
    if primary_metric == "ndcg_at_10_dev" and dev_ndcg is not None:
        return float(dev_ndcg)
    train_ndcg = result.train_metrics.get("ndcg_at_10")
    if dev_ndcg is not None:
        return float(dev_ndcg)
    if train_ndcg is not None:
        return float(train_ndcg)
    return None


def _build_metadata(
    version: int,
    run_id: str,
    cfg: TrainingConfig,
    report: DatasetReport,
    result: PipelineResult,
) -> Dict[str, Any]:
    return {
        "version": version,
        "run_id": run_id,
        "created_at": now_iso(),
        "status": "completed",
        "base_model": cfg.base_model,
        "n_packages_found": report.sources_found,
        "n_packages_used": report.sources_used,
        "n_packages_skipped": len(report.sources_skipped),
        "skipped_sources": report.sources_skipped,
        "lecture_ids": report.lecture_ids,
        "n_triplets_merged": report.n_triplets_after_dedupe,
        "n_train_triplets": result.n_train_triplets,
        "n_dev_triplets": result.n_dev_triplets,
        "hyperparameters": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "warmup_ratio": cfg.warmup_ratio,
            "evaluation_steps": cfg.evaluation_steps,
            "dev_fraction": cfg.dev_fraction,
            "seed": cfg.seed,
            "min_triplets": cfg.min_triplets,
            "max_per_lecture": cfg.max_per_lecture,
            "min_total_triplets": cfg.min_total_triplets,
        },
        "metrics": {
            "train": result.train_metrics,
            "dev": result.dev_metrics,
        },
        "eval_error": result.eval_error,
    }


# ---------------------------------------------------------------------------
# CLI (Kaggle Notebook B entry point)
# ---------------------------------------------------------------------------


def main() -> None:
    # CLI defaults honor the RERANKER_TRAINING_* env contract (set by
    # run_kaggle_reranker_training.py or .env); explicit flags win.
    env_packages = os.environ.get("RERANKER_TRAINING_PACKAGES_ROOT", "").strip()
    env_models = os.environ.get("RERANKER_TRAINING_MODELS_ROOT", "").strip()
    env_scratch = os.environ.get("RERANKER_TRAINING_SCRATCH_ROOT", "").strip()

    parser = argparse.ArgumentParser(
        description="Pipeline B — Global Reranker Training (Kaggle)."
    )
    parser.add_argument(
        "--packages-root", type=Path,
        default=Path(env_packages) if env_packages else Path(DEFAULT_PACKAGES_ROOT),
    )
    parser.add_argument(
        "--models-root", type=Path,
        default=Path(env_models) if env_models else Path(DEFAULT_MODELS_ROOT),
    )
    parser.add_argument(
        "--scratch-root", type=Path,
        default=Path(env_scratch) if env_scratch else Path(DEFAULT_SCRATCH_ROOT),
    )
    parser.add_argument("--min-triplets", type=int, default=0)
    parser.add_argument("--max-per-lecture", type=int, default=None)
    parser.add_argument("--min-total-triplets", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--eval-frequency", type=int, default=0,
                        help="Evaluate on dev triplets every N training steps (0 = off).")
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Skip the CUDA gate (debug/test only).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # GPU-only training — no CPU fallback.
    if not args.allow_cpu:
        try:
            import torch

            if not torch.cuda.is_available():
                logger.error(
                    "Pipeline B: CUDA GPU required for reranker training but "
                    "torch.cuda.is_available() is False. Aborting. "
                    "Run with --allow-cpu only for local debug."
                )
                sys.exit(1)
        except ImportError:
            logger.error("Pipeline B: torch not installed. Aborting.")
            sys.exit(1)

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        evaluation_steps=args.eval_frequency,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
        min_triplets=args.min_triplets,
        max_per_lecture=args.max_per_lecture,
        min_total_triplets=args.min_total_triplets,
        base_model=args.base_model,
    )

    result = run_training_pipeline(
        packages_root=args.packages_root,
        models_root=args.models_root,
        scratch_root=args.scratch_root,
        config=config,
        exclude_dirs=list(CloudSettings().RERANKER_TRAINING_EXCLUDE_DIRS),
    )

    print(json.dumps(asdict(result), indent=2, default=str))
    if result.status != "completed":
        sys.exit(1)


if __name__ == "__main__":
    main()
