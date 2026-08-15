# scripts/train_global_reranker.py
# Feature 2 — Global Reranker fine-tuning orchestration.
#
# The training pipeline already exists (cloud/training/triplet_generator.py,
# cloud/training/reranker_trainer.py) but nothing invokes it. This script is
# the missing link: merge triplets from imported knowledge packages → fine-tune
# bge-reranker-base → install into local_runtime/models/global_reranker/ →
# hot-reload the running server's singleton.
#
# Inference code is NOT changed — app.py already loads the global dir at
# startup, and reload_global_reranker() swaps the singleton live.
#
# Usage:
#   .venv/bin/python scripts/train_global_reranker.py                 # all packages
#   .venv/bin/python scripts/train_global_reranker.py --lectures lecture_a lecture_b
#   .venv/bin/python scripts/train_global_reranker.py --min-triplets 50 --dry-run
#   .venv/bin/python scripts/train_global_reranker.py --epochs 3 --batch-size 8 --reload

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as a script without package-installed cwd assumptions.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CloudSettings, cloud_settings, local_settings  # noqa: E402

logger = logging.getLogger("train_global_reranker")

TRAIN_LECTURE_ID = "global_reranker_train"
DEFAULT_PACKAGES_DIR = PROJECT_ROOT / "data" / "packages"


# ---------------------------------------------------------------------------
# Step 1 — collect & merge triplets
# ---------------------------------------------------------------------------

def collect_and_merge_triplets(
    packages_dir: Path,
    lecture_ids: Optional[List[str]] = None,
    min_triplets: int = 50,
    max_per_lecture: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Merge triplets.json across packages.

    Args:
        packages_dir: Directory containing lecture_* package directories.
        lecture_ids: Optional explicit list of lecture ids. Defaults to all
            packages that contain a triplets.json.
        min_triplets: Skip lectures with fewer triplets than this (their tiny
            sets add noise). Use 0 to include everything.
        max_per_lecture: Optional cap per lecture (keeps one lecture from
            dominating the merged set).

    Returns:
        Merged list of validated triplet dicts.
    """
    from schemas.triplet import RerankerTriplet

    merged: List[Dict[str, Any]] = []
    packages_dir = Path(packages_dir)
    if not packages_dir.is_dir():
        logger.warning("Packages dir not found: %s", packages_dir)
        return merged

    candidates = lecture_ids if lecture_ids is not None else [
        d.name for d in sorted(packages_dir.iterdir())
        if d.is_dir() and (d / "triplets.json").exists()
    ]

    for lid in candidates:
        trip_path = packages_dir / lid / "triplets.json"
        if not trip_path.exists():
            logger.info("Skip %s — no triplets.json", lid)
            continue
        try:
            data = json.loads(trip_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skip %s — unreadable triplets.json: %s", lid, exc)
            continue

        # Validate against the closed schema; drop malformed entries.
        valid = []
        for t in data:
            try:
                RerankerTriplet(**t)
                valid.append(t)
            except Exception:
                continue

        if len(valid) < min_triplets:
            logger.info("Skip %s — only %d valid triplets (< %d)", lid, len(valid), min_triplets)
            continue
        if max_per_lecture is not None:
            valid = valid[:max_per_lecture]
        logger.info("Merge %s — %d triplets", lid, len(valid))
        merged.extend(valid)

    logger.info("Merged %d triplets from %d package(s).", len(merged), len(candidates))
    return merged


def prepare_train_dir(
    train_id: str,
    triplets: List[Dict[str, Any]],
    settings: Optional[CloudSettings] = None,
) -> Path:
    """Write merged triplets into the trainer's expected lecture dir."""
    settings = settings or cloud_settings
    lecture_dir = Path(settings.lecture_dir(train_id))
    lecture_dir.mkdir(parents=True, exist_ok=True)
    (lecture_dir / "triplets.json").write_text(
        json.dumps(triplets, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %d triplets to %s", len(triplets), lecture_dir / "triplets.json")
    return lecture_dir


# ---------------------------------------------------------------------------
# Step 2 — train (reuses the existing trainer verbatim)
# ---------------------------------------------------------------------------

def train_global(
    train_id: str,
    triplets: List[Dict[str, Any]],
    trainer_fn: Optional[Any] = None,
    base_model: str = "BAAI/bge-reranker-base",
    epochs: int = 3,
    batch_size: int = 8,
    settings: Optional[CloudSettings] = None,
) -> Dict[str, Any]:
    """Fine-tune the reranker on the merged triplets. Returns metrics dict."""
    from cloud.training.reranker_trainer import train_reranker

    if not triplets:
        raise ValueError("No triplets to train on.")
    prepare_train_dir(train_id, triplets, settings)
    metrics = train_reranker(
        train_id,
        cloud_settings=settings,
        trainer_fn=trainer_fn,
        base_model=base_model,
        epochs=epochs,
        batch_size=batch_size,
    )
    logger.info("Training metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Step 3 — install into the global dir (atomic-ish swap)
# ---------------------------------------------------------------------------

def install_global_model(
    train_dir: Path,
    global_dir: Path,
) -> int:
    """
    Copy the trained model artifacts into GLOBAL_RERANKER_DIR.

    Copies into a sibling temp dir first, then swaps — so a partially written
    model is never visible to the running server (app.py validates config.json
    before loading).
    """
    model_src = Path(train_dir) / "reranker_model"
    if not model_src.is_dir():
        raise FileNotFoundError(f"Trained model not found at {model_src}")

    global_dir = Path(global_dir)
    global_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = global_dir.parent / f".{global_dir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(model_src, staging)
    if global_dir.exists():
        shutil.rmtree(global_dir)
    shutil.move(str(staging), str(global_dir))

    copied = len(list(global_dir.iterdir()))
    logger.info("Installed %d model files into %s", copied, global_dir)
    return copied


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune and install the global reranker.")
    parser.add_argument("--packages-dir", type=Path, default=DEFAULT_PACKAGES_DIR)
    parser.add_argument("--lectures", nargs="*", help="Explicit lecture ids (default: all with triplets).")
    parser.add_argument("--min-triplets", type=int, default=50)
    parser.add_argument("--max-per-lecture", type=int, default=None)
    parser.add_argument("--base-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Merge only; report what would train.")
    parser.add_argument("--no-install", action="store_true", help="Train but do not install/reload.")
    parser.add_argument("--reload", action="store_true", help="Hot-reload the running server after install.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    triplets = collect_and_merge_triplets(
        args.packages_dir,
        lecture_ids=args.lectures,
        min_triplets=args.min_triplets,
        max_per_lecture=args.max_per_lecture,
    )
    if not triplets:
        logger.error("Nothing to train on. Aborting.")
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN — would fine-tune on %d triplets (epochs=%d, batch=%d).", len(triplets), args.epochs, args.batch_size)
        return

    metrics = train_global(
        TRAIN_LECTURE_ID,
        triplets,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    if args.no_install:
        logger.info("Training done (metrics above); skipping install.")
        return

    train_dir = Path(cloud_settings.lecture_dir(TRAIN_LECTURE_ID))
    install_global_model(train_dir, Path(local_settings.GLOBAL_RERANKER_DIR))

    if args.reload:
        from retrieval.reranker.rerank_service import reload_global_reranker
        reload_global_reranker(str(local_settings.GLOBAL_RERANKER_DIR))
        logger.info("Global reranker singleton hot-reloaded.")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
