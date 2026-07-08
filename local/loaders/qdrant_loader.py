# local/loaders/qdrant_loader.py
# D2 — Qdrant Vector Store Loader.
#
# Loads embeddings.npy + metadata into a Qdrant collection.
# Collection: vector_size=1024, distance=Cosine.
# Point ID derived from chunk_id (deterministic UUID5).
#
# Payload contains exactly: lecture_id, chunk_id, segment_id,
# timestamp, transcript, visual_context, ocr_text.
# No additional fields.
#
# Environment: Local only. Never imports from cloud/.

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import LocalSettings, SharedSettings

logger = logging.getLogger(__name__)

# UUID namespace for deterministic point IDs from chunk_id.
QDRANT_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
COLLECTION_NAME = "lecturemind_chunks"


class QdrantLoadError(Exception):
    """Raised when Qdrant loading fails."""
    pass


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUID from chunk_id for Qdrant point ID."""
    return str(uuid.uuid5(QDRANT_NAMESPACE, chunk_id))


def _build_chunk_segment_map(
    package_dir: Path,
) -> Dict[str, str]:
    """
    Build chunk_id → segment_id mapping.

    Uses chunk_segment_map.json if available, falls back to segments.json.
    """
    csm_path = package_dir / "chunk_segment_map.json"
    if csm_path.exists():
        raw = json.loads(csm_path.read_text(encoding="utf-8"))
        # chunk_segment_map.json is a flat dict: { chunk_id: segment_id }
        if isinstance(raw, dict):
            # Could be raw dict or wrapped in ChunkSegmentMap schema.
            if "mapping" in raw:
                return raw["mapping"]
            return raw

    # Fallback: derive from segments.json.
    segments_path = package_dir / "segments.json"
    if not segments_path.exists():
        raise QdrantLoadError("Neither chunk_segment_map.json nor segments.json found.")

    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    mapping = {}
    for seg in segments:
        seg_id = seg["segment_id"]
        for cid in seg.get("chunks", []):
            mapping[cid] = seg_id

    return mapping


def _build_payloads(
    package_dir: Path,
    embedding_ids: List[Dict],
    chunk_segment_map: Dict[str, str],
    lecture_id: str,
) -> List[Dict[str, Any]]:
    """
    Build Qdrant payloads for each embedding vector.

    Payload fields: lecture_id, chunk_id, segment_id, timestamp,
    transcript, visual_context, ocr_text.

    lecture_id is the runtime upload ID (e.g. "lecture_b3fb8ab8"),
    NOT the cloud-generated ID from the package (e.g. "lec_001").
    This ensures rerank_service can look up the package via LectureRegistry.
    """
    chunks_path = package_dir / "multimodal_chunks.json"
    if not chunks_path.exists():
        raise QdrantLoadError("multimodal_chunks.json not found.")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunk_lookup = {c["chunk_id"]: c for c in chunks}

    payloads = []
    for entry in embedding_ids:
        chunk_id = entry["chunk_id"]
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            raise QdrantLoadError(f"chunk_id '{chunk_id}' in embedding_ids not found in chunks.")

        segment_id = chunk_segment_map.get(chunk_id, "")
        if not segment_id:
            logger.warning("D2: No segment_id found for chunk '%s'.", chunk_id)

        payload = {
            "lecture_id": lecture_id,
            "chunk_id": chunk_id,
            "segment_id": segment_id,
            "timestamp": chunk["timestamp"],
            "transcript": chunk["transcript"],
            "visual_context": chunk["visual_context"],
            "ocr_text": chunk["ocr_text"],
        }
        payloads.append(payload)

    return payloads


def load_qdrant(
    package_dir: Path,
    lecture_id: str = "",
    local_settings: LocalSettings | None = None,
    shared_settings: SharedSettings | None = None,
    qdrant_client: Optional[Any] = None,
    collection_name: str = COLLECTION_NAME,
) -> Dict[str, Any]:
    """
    Load embeddings and metadata into Qdrant.

    Args:
        package_dir: Path to the unpacked knowledge package.
        lecture_id: Runtime upload lecture ID (e.g. "lecture_b3fb8ab8").
                    Stored in every Qdrant payload so rerank_service can
                    resolve the package via LectureRegistry. Defaults to
                    the cloud-internal ID if not supplied (testing only).
        local_settings: Injected LocalSettings.
        shared_settings: SharedSettings for EMBEDDING_DIMENSION.
        qdrant_client: Optional pre-created Qdrant client (for testing).
        collection_name: Qdrant collection name.

    Returns:
        Summary dict with point count and collection info.
    """
    settings = local_settings or LocalSettings()
    ss = shared_settings or SharedSettings()
    package_dir = Path(package_dir)

    # Load embeddings.
    emb_path = package_dir / "embeddings.npy"
    ids_path = package_dir / "embedding_ids.json"

    if not emb_path.exists():
        raise QdrantLoadError(f"embeddings.npy not found: {emb_path}")
    if not ids_path.exists():
        raise QdrantLoadError(f"embedding_ids.json not found: {ids_path}")

    embeddings = np.load(str(emb_path))
    embedding_ids = json.loads(ids_path.read_text(encoding="utf-8"))

    # Validate dimension.
    if embeddings.ndim != 2 or embeddings.shape[1] != ss.EMBEDDING_DIMENSION:
        raise QdrantLoadError(
            f"Embedding dimension mismatch: expected (N, {ss.EMBEDDING_DIMENSION}), "
            f"got {embeddings.shape}"
        )

    if embeddings.shape[0] != len(embedding_ids):
        raise QdrantLoadError(
            f"Row count mismatch: embeddings has {embeddings.shape[0]} rows, "
            f"embedding_ids has {len(embedding_ids)} entries."
        )

    # Build chunk→segment map.
    chunk_segment_map = _build_chunk_segment_map(package_dir)

    # If no runtime lecture_id was supplied, fall back to the package-internal ID
    # (only reached in legacy tests; upload route always passes lecture_id).
    resolved_lecture_id = lecture_id
    if not resolved_lecture_id:
        import json as _json  # already imported; reference for clarity
        first_chunk_path = package_dir / "multimodal_chunks.json"
        _chunks = _json.loads(first_chunk_path.read_text(encoding="utf-8"))
        resolved_lecture_id = _chunks[0]["lecture_id"] if _chunks else ""
        logger.warning(
            "D2: No lecture_id supplied to load_qdrant(); "
            "falling back to package-internal ID '%s'. "
            "Pass lecture_id= in production.",
            resolved_lecture_id,
        )

    # Build payloads.
    payloads = _build_payloads(package_dir, embedding_ids, chunk_segment_map, resolved_lecture_id)

    # Connect to Qdrant.
    close_client = False
    use_sdk = False
    if qdrant_client is None:
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(url=settings.QDRANT_URL)
        close_client = True
        use_sdk = True

    try:
        # Create collection if it doesn't exist.
        if use_sdk:
            from qdrant_client.models import VectorParams, Distance
            if not qdrant_client.collection_exists(collection_name=collection_name):
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=ss.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE,
                    ),
                )
        else:
            # Mock-compatible fallback.
            try:
                cols = qdrant_client.get_collections()
                names = [c.name for c in cols.collections] if hasattr(cols, "collections") else []
                if collection_name not in names:
                    qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config={"size": ss.EMBEDDING_DIMENSION, "distance": "Cosine"},
                    )
            except Exception:
                qdrant_client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config={"size": ss.EMBEDDING_DIMENSION, "distance": "Cosine"},
                )

        logger.info("D2: Collection '%s' checked/created (dim=%d, Cosine).", collection_name, ss.EMBEDDING_DIMENSION)

        # Build points as simple objects with id/vector/payload.
        _Point = None
        if use_sdk:
            from qdrant_client.models import PointStruct
            _Point = PointStruct

        points = []
        for i, entry in enumerate(embedding_ids):
            chunk_id = entry["chunk_id"]
            point_id = _chunk_id_to_uuid(chunk_id)
            vector = embeddings[entry["row_index"]].tolist()
            payload = payloads[i]

            if _Point is not None:
                points.append(_Point(id=point_id, vector=vector, payload=payload))
            else:
                from types import SimpleNamespace
                points.append(SimpleNamespace(id=point_id, vector=vector, payload=payload))

        # Upsert in batches.
        batch_size = 100
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            qdrant_client.upsert(collection_name=collection_name, points=batch)

        logger.info("D2: Upserted %d points into '%s'.", len(points), collection_name)

    finally:
        if close_client:
            qdrant_client.close()

    result = {
        "collection_name": collection_name,
        "point_count": len(points),
        "vector_dimension": ss.EMBEDDING_DIMENSION,
    }
    logger.info("D2: Qdrant load complete — %s", result)
    return result
