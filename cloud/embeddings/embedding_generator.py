# cloud/embeddings/embedding_generator.py
# Stage B0 — Embedding Generation.
#
# Encodes multimodal chunks using BAAI/bge-large-en-v1.5 (1024-dim).
# Concatenates transcript + visual_context + ocr_text per chunk.
#
# Output: cloud_runtime/lectures/{lecture_id}/embeddings.npy
#         cloud_runtime/lectures/{lecture_id}/embedding_ids.json
# Environment: Kaggle GPU only.

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from config import CloudSettings, SharedSettings
from schemas.chunk import MultimodalChunk

logger = logging.getLogger(__name__)

# Get local Kaggle model path (GPU memory optimization)
_settings = CloudSettings()
EMBEDDING_MODEL = _settings.BGE_MODEL_PATH


def _load_embedding_model(model_name: str = EMBEDDING_MODEL) -> Any:
    """
    Load sentence-transformers embedding model from local Kaggle path.

    Separated for test mockability.
    """
    from sentence_transformers import SentenceTransformer
    from pathlib import Path

    if not Path(model_name).exists():
        raise FileNotFoundError(f"Model path does not exist: {model_name}")

    model = SentenceTransformer(model_name)
    logger.info("B0: Loaded embedding model '%s' from local path.", model_name)
    return model


def _build_texts(chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Build combined text for each chunk: transcript + visual_context + ocr_text.

    No additional fields are read or invented.
    """
    texts = []
    for chunk in chunks:
        combined = (
            chunk["transcript"]
            + "\n"
            + chunk["visual_context"]
            + "\n"
            + chunk["ocr_text"]
        )
        texts.append(combined)
    return texts


def generate_embeddings(
    lecture_id: str,
    cloud_settings: CloudSettings | None = None,
    shared_settings: SharedSettings | None = None,
    model_loader: Any = None,
) -> int:
    """
    Generate embeddings for all multimodal chunks and save as .npy + mapping.

    Args:
        lecture_id: Unique lecture identifier.
        cloud_settings: Injected CloudSettings.
        shared_settings: Injected SharedSettings (for EMBEDDING_DIMENSION).
        model_loader: Optional callable returning an embedding model for testing.

    Returns:
        Number of chunks embedded.

    Raises:
        FileNotFoundError: If multimodal_chunks.json is missing.
        ValueError: If embedding dimension != 1024.
    """
    cs = cloud_settings or CloudSettings()
    ss = shared_settings or SharedSettings()
    lecture_dir = Path(cs.lecture_dir(lecture_id))

    # Read multimodal_chunks.json (A6 output).
    chunks_path = lecture_dir / "multimodal_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"B0: multimodal_chunks.json not found: {chunks_path}")

    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))

    # Validate each chunk against Chunk 1 schema.
    for chunk in chunks_data:
        MultimodalChunk(**chunk)

    if not chunks_data:
        raise ValueError("B0: multimodal_chunks.json contains zero chunks.")

    # Build combined texts.
    texts = _build_texts(chunks_data)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks_data]

    # Load model and encode.
    if model_loader is not None:
        model = model_loader()
    else:
        model = _load_embedding_model()

    logger.info("B0: Encoding %d chunks...", len(texts))
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Validate dimension.
    if embeddings.shape[1] != ss.EMBEDDING_DIMENSION:
        raise ValueError(
            f"B0: Embedding dimension mismatch. Expected {ss.EMBEDDING_DIMENSION}, "
            f"got {embeddings.shape[1]}."
        )

    logger.info("B0: Embeddings shape: %s", embeddings.shape)

    # Save embeddings.npy
    embeddings_path = lecture_dir / "embeddings.npy"
    np.save(str(embeddings_path), embeddings)

    # Save embedding_ids.json — explicit row_index → chunk_id mapping.
    embedding_ids = [
        {"row_index": idx, "chunk_id": cid}
        for idx, cid in enumerate(chunk_ids)
    ]
    ids_path = lecture_dir / "embedding_ids.json"
    ids_path.write_text(
        json.dumps(embedding_ids, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "B0: embeddings.npy + embedding_ids.json written (%d vectors, dim=%d) to %s",
        len(chunk_ids), embeddings.shape[1], lecture_dir,
    )

    return len(chunk_ids)
