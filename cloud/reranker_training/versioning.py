# cloud/reranker_training/versioning.py
#
# Pipeline B, Step 4 — versioned model store.
#
#   reranker_models/
#     index.json          # registry: versions list, latest, best + best metric
#     v1/
#       model/            # CrossEncoder artifacts (config.json, model.safetensors, ...)
#       metadata.json     # version provenance (date, lectures, triplets, hyperparams)
#       evaluation_report.json
#     v2/ ...#   best/               # copy of the best-performing version (never overwritten
#                         #   without a strict comparison)
#   latest/             # copy of the newest version
#
# Commit policy:
#   * a version directory is created ONLY after training succeeded (atomic
#     staging + rename), so an interrupted run can never leave a half-written v{N}
#   * best/ is replaced ONLY when the new primary metric is strictly greater
#     than the stored best metric — ties and regressions keep the old best
#
# The store is SINGLE-WRITER: one training run at a time (the Kaggle notebook
# is a single process). Staging-dir names are unique per operation, so a
# crash mid-commit leaves only a stale .staging/ the next run cleans up.

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.json"
METADATA_FILENAME = "metadata.json"
EVAL_REPORT_FILENAME = "evaluation_report.json"
MODEL_SUBDIR = "model"
VERSION_DIR_RE = re.compile(r"^v(\d+)$")

PRIMARY_METRIC_DEFAULT = "ndcg_at_10_dev"


def next_version(models_root: Path) -> int:
    """Highest existing v{N} + 1. Scans the filesystem (source of truth)."""
    models_root = Path(models_root)
    highest = 0
    if models_root.is_dir():
        for entry in models_root.iterdir():
            m = VERSION_DIR_RE.match(entry.name)
            if m and entry.is_dir():
                highest = max(highest, int(m.group(1)))
    return highest + 1


def load_index(models_root: Path) -> Dict[str, Any]:
    path = Path(models_root) / INDEX_FILENAME
    if not path.is_file():
        return {"versions": [], "latest": None, "best": None, "best_metric": None,
                "primary_metric": PRIMARY_METRIC_DEFAULT}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("versions", [])
        data.setdefault("primary_metric", PRIMARY_METRIC_DEFAULT)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Pipeline B: unreadable index.json (%s) — rebuilding.", exc)
        return {"versions": [], "latest": None, "best": None, "best_metric": None,
                "primary_metric": PRIMARY_METRIC_DEFAULT}


def save_index(models_root: Path, index: Dict[str, Any]) -> None:
    path = Path(models_root) / INDEX_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def commit_version(
    models_root: Path,
    version: int,
    model_src: Path,
    metadata: Dict[str, Any],
    evaluation_report: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Atomically commit a trained model as reranker_models/v{N}/.

    Copies model_src → staging → rename, so a crash mid-copy never leaves a
    partial v{N} visible. Returns the committed version directory.
    """
    models_root = Path(models_root)
    model_src = Path(model_src)
    if not model_src.is_dir():
        raise FileNotFoundError(f"Trained model dir not found: {model_src}")

    models_root.mkdir(parents=True, exist_ok=True)
    version_dir = models_root / f"v{version}"
    staging = models_root / f".v{version}.staging"

    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(model_src, staging / MODEL_SUBDIR)

    (staging / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (staging / EVAL_REPORT_FILENAME).write_text(
        json.dumps(evaluation_report or {}, indent=2), encoding="utf-8"
    )

    if version_dir.exists():
        shutil.rmtree(version_dir)
    shutil.move(str(staging), str(version_dir))

    index = load_index(models_root)
    versions = index.setdefault("versions", [])
    if version not in versions:
        versions.append(version)
        versions.sort()
    index["latest"] = version
    save_index(models_root, index)

    logger.info("Pipeline B: committed version v%d → %s", version, version_dir)
    return version_dir


def _copy_version_dir(src: Path, dst: Path) -> None:
    """Copy a full version dir (model + metadata + report) atomically."""
    dst = Path(dst)
    staging = dst.parent / f".{dst.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(src, staging)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(staging), str(dst))


def update_pointers(
    models_root: Path,
    version: int,
    primary_value: Optional[float],
    primary_metric: str = PRIMARY_METRIC_DEFAULT,
) -> Dict[str, Any]:
    """
    Update best/ and latest/ pointers after committing v{version}.

    best/ is replaced ONLY if primary_value is strictly greater than the
    current best_metric. latest/ always points at the newest version.

    Returns a dict describing the pointer decisions.
    """
    models_root = Path(models_root)
    version_dir = models_root / f"v{version}"
    if not version_dir.is_dir():
        raise FileNotFoundError(f"Version dir not found: {version_dir}")

    index = load_index(models_root)
    index["latest"] = version
    index["primary_metric"] = primary_metric
    result: Dict[str, Any] = {
        "version": version,
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "best_updated": False,
        "latest_updated": True,
    }

    current_best = index.get("best")
    current_best_metric = index.get("best_metric")

    if primary_value is not None and (
        current_best_metric is None or primary_value > float(current_best_metric)
    ):
        _copy_version_dir(version_dir, models_root / "best")
        index["best"] = version
        index["best_metric"] = primary_value
        result["best_updated"] = True
        logger.info(
            "Pipeline B: best updated to v%d (primary metric %.4f).",
            version, primary_value,
        )
    elif current_best is not None:
        logger.info(
            "Pipeline B: best unchanged — v%d primary metric %.4f is not strictly "
            "better than best v%d (%.4f).",
            version, primary_value if primary_value is not None else -1.0,
            current_best, current_best_metric,
        )
    else:
        # No previous best, and no measurable primary metric — record best anyway
        # so the store has a stable pointer (e.g. first run, eval failed).
        _copy_version_dir(version_dir, models_root / "best")
        index["best"] = version
        index["best_metric"] = None
        result["best_updated"] = True

    _copy_version_dir(version_dir, models_root / "latest")
    save_index(models_root, index)
    return result


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
