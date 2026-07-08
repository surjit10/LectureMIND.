# local/loaders/manifest_validator.py
# Local-side manifest validation for unpacked knowledge packages.
#
# Cross-checks manifest.json counts against actual artifact file contents.
# Verifies embedding_dimension == 1024.
# Any mismatch is a hard fail — no warnings.
#
# Environment: Local only. Never imports from cloud/.

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from config import LocalSettings, SharedSettings
from schemas.manifest import Manifest

logger = logging.getLogger(__name__)


class ManifestValidationError(Exception):
    """Raised when local manifest validation fails."""
    pass


def validate_manifest(
    package_dir: Path,
    shared_settings: SharedSettings | None = None,
) -> Dict[str, Any]:
    """
    Validate manifest.json against actual package contents.

    Checks:
    1. manifest.json exists and parses against Manifest schema.
    2. embedding_dimension == 1024.
    3. chunk_count matches multimodal_chunks.json length.
    4. segment_count matches segments.json length.
    5. entity_count matches entities.json length.
    6. relation_count matches relations.json length.

    Args:
        package_dir: Path to the unpacked knowledge package directory.
        shared_settings: SharedSettings for EMBEDDING_DIMENSION.

    Returns:
        Validation summary dict.

    Raises:
        ManifestValidationError: On any validation failure.
    """
    ss = shared_settings or SharedSettings()
    package_dir = Path(package_dir)

    # 1. Load and validate manifest schema.
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        raise ManifestValidationError(f"manifest.json not found in {package_dir}")

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = Manifest(**manifest_data)
    except Exception as exc:
        raise ManifestValidationError(f"manifest.json schema validation failed: {exc}")

    # 2. Embedding dimension must be 1024.
    if manifest.embedding_dimension != ss.EMBEDDING_DIMENSION:
        raise ManifestValidationError(
            f"embedding_dimension mismatch: manifest says {manifest.embedding_dimension}, "
            f"expected {ss.EMBEDDING_DIMENSION}"
        )

    # 3. Verify required files exist.
    required_files = [
        "multimodal_chunks.json", "segments.json",
        "entities.json", "relations.json",
        "embeddings.npy", "embedding_ids.json",
    ]
    for filename in required_files:
        if not (package_dir / filename).exists():
            raise ManifestValidationError(f"Required file missing: {filename}")

    # NOTE: reranker_model/ is no longer required in the package.
    # The global reranker is loaded at application startup from a fixed path.

    # 4. Cross-check counts.
    chunks = json.loads((package_dir / "multimodal_chunks.json").read_text(encoding="utf-8"))
    segments = json.loads((package_dir / "segments.json").read_text(encoding="utf-8"))
    entities = json.loads((package_dir / "entities.json").read_text(encoding="utf-8"))
    relations = json.loads((package_dir / "relations.json").read_text(encoding="utf-8"))

    if manifest.chunk_count != len(chunks):
        raise ManifestValidationError(
            f"chunk_count mismatch: manifest={manifest.chunk_count}, actual={len(chunks)}"
        )
    if manifest.segment_count != len(segments):
        raise ManifestValidationError(
            f"segment_count mismatch: manifest={manifest.segment_count}, actual={len(segments)}"
        )
    if manifest.entity_count != len(entities):
        raise ManifestValidationError(
            f"entity_count mismatch: manifest={manifest.entity_count}, actual={len(entities)}"
        )
    if manifest.relation_count != len(relations):
        raise ManifestValidationError(
            f"relation_count mismatch: manifest={manifest.relation_count}, actual={len(relations)}"
        )

    # 5. Verify embedding file dimension.
    embeddings = np.load(str(package_dir / "embeddings.npy"))
    if embeddings.ndim != 2 or embeddings.shape[1] != ss.EMBEDDING_DIMENSION:
        raise ManifestValidationError(
            f"embeddings.npy dimension mismatch: expected (N, {ss.EMBEDDING_DIMENSION}), "
            f"got {embeddings.shape}"
        )

    summary = {
        "status": "VALID",
        "lecture_id": manifest.lecture_id,
        "embedding_dimension": manifest.embedding_dimension,
        "chunk_count": manifest.chunk_count,
        "segment_count": manifest.segment_count,
        "entity_count": manifest.entity_count,
        "relation_count": manifest.relation_count,
    }
    logger.info("Manifest validation PASSED for '%s'.", manifest.lecture_id)
    return summary
