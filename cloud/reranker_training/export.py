# cloud/reranker_training/export.py
#
# Pipeline B, Step 5 — export the trained reranker only.
#
# Produces reranker_models/global_reranker_v{N}.zip containing:
#   * model weights + tokenizer + config (the CrossEncoder artifacts)
#   * metadata.json        (version provenance: date, lectures, triplets, hyperparams)
#   * evaluation_report.json (dev metrics, loss history)
#
# Never includes: lecture packages, scratch/training intermediates, staging
# dirs, logs, or anything outside the committed version dir.
#
# The zip is FLAT (model files at the archive root) so the local upload flow
# (POST /api/reranker/upload) accepts it unchanged.

import logging
import zipfile
from pathlib import Path
from typing import Optional

from cloud.reranker_training.versioning import (
    EVAL_REPORT_FILENAME,
    METADATA_FILENAME,
    MODEL_SUBDIR,
)

logger = logging.getLogger(__name__)

SKIPPED_NAMES = {".DS_Store", "__MACOSX", ".staging"}


def export_version(
    models_root: Path,
    version: int,
    out_dir: Optional[Path] = None,
) -> Path:
    """
    Zip the committed version dir into global_reranker_v{N}.zip.

    Args:
        models_root: Versioned store root (reranker_models/).
        version: Version number to export.
        out_dir: Destination directory for the zip (defaults to models_root).

    Returns:
        Path to the created zip.
    """
    models_root = Path(models_root)
    version_dir = models_root / f"v{version}"
    model_dir = version_dir / MODEL_SUBDIR
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Version model dir not found: {model_dir}")

    out = Path(out_dir) if out_dir is not None else models_root
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / f"global_reranker_v{version}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Model artifacts — flat at the archive root (subdirs preserved
        # relative, though CrossEncoder saves are flat in practice).
        for file_path in sorted(model_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if _should_skip(file_path):
                continue
            arcname = file_path.relative_to(model_dir).as_posix()
            zf.write(file_path, arcname)

        # Provenance files at the root.
        for name in (METADATA_FILENAME, EVAL_REPORT_FILENAME):
            p = version_dir / name
            if p.is_file():
                zf.write(p, name)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Pipeline B: exported %s (%.1f MB).", zip_path.name, size_mb,
    )
    return zip_path


def _should_skip(path: Path) -> bool:
    """Skip temp/hidden/junk files inside the model dir."""
    if path.name.startswith("."):
        return True
    return any(part in SKIPPED_NAMES for part in path.parts)
