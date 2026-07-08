# cloud/packaging/manifest_builder.py
# Manifest Generation — builds manifest.json from pipeline outputs.
#
# Input:  multimodal_chunks.json, segments.json, entities.json, relations.json
# Output: cloud_runtime/lectures/{lecture_id}/manifest.json
# Environment: Kaggle GPU only.
#
# CRITICAL: Output must validate against Manifest schema.
# embedding_dimension MUST be 1024.

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from config import CloudSettings, SharedSettings
from schemas.manifest import Manifest

logger = logging.getLogger(__name__)

# Use local Kaggle model paths (GPU memory optimization)
_settings = CloudSettings()
EMBEDDING_MODEL = _settings.BGE_MODEL_PATH
RERANKER_VERSION = "BAAI/bge-reranker-base"


def build_manifest(
    lecture_id: str,
    cloud_settings: CloudSettings | None = None,
    shared_settings: SharedSettings | None = None,
    embedding_model: str = EMBEDDING_MODEL,
    reranker_version: str = RERANKER_VERSION,
    # V7 optional metadata
    display_name: str | None = None,
    course_name: str | None = None,
    speaker: str | None = None,
    description: str | None = None,
    language: str | None = None,
    duration: float | None = None,
    pipeline_version: str | None = None,
    package_version: str | None = None,
) -> Manifest:
    """
    Build and write manifest.json from all pipeline artifacts.

    Args:
        lecture_id: Unique lecture identifier.
        cloud_settings: Injected CloudSettings.
        shared_settings: Injected SharedSettings (for EMBEDDING_DIMENSION).
        embedding_model: Name of the embedding model used.
        reranker_version: Version string for the reranker model.
        display_name: Human-readable lecture title (V7 optional).
        course_name: Course or module name (V7 optional).
        speaker: Lecturer / speaker name (V7 optional).
        description: Short description of lecture content (V7 optional).
        language: BCP-47 language code, e.g. "en" (V7 optional).
        duration: Lecture duration in seconds (V7 optional).
        pipeline_version: Cloud pipeline version tag (V7 optional).
        package_version: Knowledge package format version (V7 optional).

    Returns:
        Validated Manifest instance.

    Raises:
        FileNotFoundError: If any required input file is missing.
    """
    cs = cloud_settings or CloudSettings()
    ss = shared_settings or SharedSettings()
    lecture_dir = Path(cs.lecture_dir(lecture_id))

    # Load and count artifacts.
    chunks_path = lecture_dir / "multimodal_chunks.json"
    segments_path = lecture_dir / "segments.json"
    entities_path = lecture_dir / "entities.json"
    relations_path = lecture_dir / "relations.json"

    if not chunks_path.exists():
        raise FileNotFoundError(f"Manifest: multimodal_chunks.json not found: {chunks_path}")
    if not segments_path.exists():
        raise FileNotFoundError(f"Manifest: segments.json not found: {segments_path}")
    if not entities_path.exists():
        raise FileNotFoundError(f"Manifest: entities.json not found: {entities_path}")
    if not relations_path.exists():
        raise FileNotFoundError(f"Manifest: relations.json not found: {relations_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    entities = json.loads(entities_path.read_text(encoding="utf-8"))
    relations = json.loads(relations_path.read_text(encoding="utf-8"))

    # Build manifest.
    manifest = Manifest(
        lecture_id=lecture_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        embedding_model=embedding_model,
        embedding_dimension=ss.EMBEDDING_DIMENSION,
        chunk_count=len(chunks),
        segment_count=len(segments),
        entity_count=len(entities),
        relation_count=len(relations),
        reranker_version=reranker_version,
        # V7 optional metadata
        display_name=display_name,
        course_name=course_name,
        speaker=speaker,
        description=description,
        language=language,
        duration=duration,
        pipeline_version=pipeline_version,
        package_version=package_version,
    )

    # Write manifest.json.
    # exclude_none=True: V7 optional fields absent from the call are omitted
    # from the output file, preserving backward compatibility with validators
    # that check for exactly the 9 core fields.
    manifest_path = lecture_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(exclude_none=True), indent=2),
        encoding="utf-8",
    )
    logger.info("Manifest: manifest.json written to %s", manifest_path)
    logger.info(
        "Manifest: chunks=%d, segments=%d, entities=%d, relations=%d",
        manifest.chunk_count, manifest.segment_count,
        manifest.entity_count, manifest.relation_count,
    )

    return manifest
