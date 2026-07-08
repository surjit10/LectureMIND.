# cloud/tests/test_b0_embeddings.py
# Tests for Stage B0 — Embedding Generation.
# Model is mocked — no downloads.

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from config import CloudSettings, SharedSettings


@pytest.fixture
def cloud_settings(tmp_path):
    return CloudSettings(
        LECTURE_OUTPUT_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/"),
        FRAME_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/frames/"),
        LOG_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/logs/"),
    )


@pytest.fixture
def setup_chunks(cloud_settings):
    """Create a minimal multimodal_chunks.json."""
    lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
    lecture_dir.mkdir(parents=True, exist_ok=True)

    chunks = [
        {
            "lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000001",
            "timestamp": 10.0, "transcript": "BFS uses queue",
            "visual_context": "BFS diagram", "ocr_text": "O(V+E)",
        },
        {
            "lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000002",
            "timestamp": 20.0, "transcript": "DFS uses stack",
            "visual_context": "DFS tree", "ocr_text": "O(V+E)",
        },
        {
            "lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000003",
            "timestamp": 30.0, "transcript": "Dijkstra shortest path",
            "visual_context": "Graph weights", "ocr_text": "O(E log V)",
        },
    ]
    (lecture_dir / "multimodal_chunks.json").write_text(json.dumps(chunks))
    return lecture_dir


def _make_mock_model(dim=1024):
    """Create a mock embedding model returning vectors of the right dimension."""
    mock = MagicMock()
    mock.encode.side_effect = lambda texts, **kwargs: np.random.randn(len(texts), dim).astype(np.float32)
    return mock


class TestEmbeddingGenerator:

    def test_generate_embeddings_success(self, cloud_settings, setup_chunks):
        """Produces embeddings.npy [N, 1024] and embedding_ids.json."""
        from cloud.embeddings.embedding_generator import generate_embeddings

        count = generate_embeddings(
            "lec_001",
            cloud_settings=cloud_settings,
            model_loader=lambda: _make_mock_model(1024),
        )

        assert count == 3
        lecture_dir = setup_chunks

        # Check embeddings.npy.
        emb = np.load(str(lecture_dir / "embeddings.npy"))
        assert emb.shape == (3, 1024)

        # Check embedding_ids.json.
        ids = json.loads((lecture_dir / "embedding_ids.json").read_text())
        assert len(ids) == 3
        assert ids[0]["row_index"] == 0
        assert ids[0]["chunk_id"] == "lec_001_chunk_000001"

    def test_wrong_dimension_rejected(self, cloud_settings, setup_chunks):
        """Dimension != 1024 must raise ValueError."""
        from cloud.embeddings.embedding_generator import generate_embeddings

        with pytest.raises(ValueError, match="1024"):
            generate_embeddings(
                "lec_001",
                cloud_settings=cloud_settings,
                model_loader=lambda: _make_mock_model(768),
            )

    def test_missing_chunks_raises(self, cloud_settings):
        """Missing multimodal_chunks.json raises FileNotFoundError."""
        from cloud.embeddings.embedding_generator import generate_embeddings

        with pytest.raises(FileNotFoundError):
            generate_embeddings("lec_001", cloud_settings=cloud_settings, model_loader=lambda: MagicMock())
