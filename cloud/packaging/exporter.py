# cloud/packaging/exporter.py
# Stage C2 — Knowledge Package Export ★ HANDOFF BOUNDARY.
#
# Zips selected artifacts from cloud_runtime into transfer/knowledge_package.zip.
# Only includes files needed for local loading.
#
# NEVER includes: logs/, frames/, vlm_output.jsonl, ocr_output.jsonl,
#                 transcript.json, frames.json, metadata.json, triplets.json.
#
# Environment: Kaggle only.

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Dict

from config import CloudSettings
from cloud.packaging.validator import validate_package

logger = logging.getLogger(__name__)

# Files included in the knowledge package.
PACKAGE_FILES = [
    "manifest.json",
    "chunk_segment_map.json",
    "entities.json",
    "relations.json",
    "segments.json",
    "multimodal_chunks.json",
    "embeddings.npy",
    "embedding_ids.json",
]

# Directories included in the knowledge package.
# NOTE: reranker_model/ is intentionally excluded — the global reranker
# is trained separately and loaded at local application startup.
PACKAGE_DIRS: list = []

# Files NEVER included (cloud-side only).
EXCLUDED_FILES = [
    "vlm_output.jsonl",
    "ocr_output.jsonl",
    "transcript.json",
    "frames.json",
    "metadata.json",
    "triplets.json",
    "training_metrics.json",
]


def export_package(
    lecture_id: str,
    cloud_settings: CloudSettings | None = None,
    transfer_dir: str = "transfer",
    skip_validation: bool = False,
) -> Path:
    """
    Export the knowledge package as a zip file.

    1. Runs C1 validation (unless skipped).
    2. Zips only the allowed artifacts.
    3. Places the zip in transfer/.

    Args:
        lecture_id: Unique lecture identifier.
        cloud_settings: Injected CloudSettings.
        transfer_dir: Directory for the output zip.
        skip_validation: Skip C1 validation (for testing only).

    Returns:
        Path to the created knowledge_package.zip.

    Raises:
        ValidationError: If C1 validation fails.
    """
    settings = cloud_settings or CloudSettings()
    lecture_dir = Path(settings.lecture_dir(lecture_id))

    # Step 1: Validate (C1).
    if not skip_validation:
        logger.info("C2: Running C1 validation before export...")
        validate_package(lecture_id, cloud_settings=settings)
        logger.info("C2: Validation passed.")

    # Step 2: Create transfer directory.
    transfer_path = Path(transfer_dir)
    transfer_path.mkdir(parents=True, exist_ok=True)
    zip_path = transfer_path / f"{lecture_id}_knowledge_package.zip"

    # Step 3: Build zip.
    logger.info("C2: Building knowledge_package.zip...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add individual files.
        for filename in PACKAGE_FILES:
            filepath = lecture_dir / filename
            if filepath.exists():
                arcname = f"knowledge_package/{filename}"
                zf.write(filepath, arcname)
                logger.debug("C2: Added %s", arcname)
            else:
                # chunk_segment_map.json is optional.
                if filename != "chunk_segment_map.json":
                    logger.warning("C2: Expected file missing from package: %s", filename)

        # Add directories.
        for dirname in PACKAGE_DIRS:
            dirpath = lecture_dir / dirname
            if dirpath.is_dir():
                for file in dirpath.rglob("*"):
                    if file.is_file():
                        arcname = f"knowledge_package/{dirname}/{file.relative_to(dirpath)}"
                        zf.write(file, arcname)
                logger.debug("C2: Added directory %s/", dirname)

    # Verify the zip was created.
    if not zip_path.exists():
        raise RuntimeError(f"C2: Failed to create {zip_path}")

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info("C2: knowledge_package.zip created (%.1f MB) at %s", zip_size_mb, zip_path)

    # Verify nothing excluded leaked in.
    _verify_exclusions(zip_path)

    return zip_path


def _verify_exclusions(zip_path: Path) -> None:
    """Verify no excluded files are in the zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for excluded in EXCLUDED_FILES:
            for name in names:
                if name.endswith(excluded):
                    logger.error("C2: EXCLUDED file found in package: %s", name)
                    raise RuntimeError(
                        f"C2: Excluded file '{excluded}' leaked into knowledge_package.zip"
                    )
        # Also check for excluded directories.
        for name in names:
            if "/frames/" in name or "/logs/" in name:
                raise RuntimeError(
                    f"C2: Excluded directory content found in package: {name}"
                )
