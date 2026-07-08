# cloud/tests/test_a4_visual_understanding.py
# Tests for Stage A4 — Qwen2-VL Visual Understanding.
# All model calls are mocked — no downloads.

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from schemas.vlm import VLMCaption
from config import CloudSettings


@pytest.fixture
def cloud_settings(tmp_path):
    return CloudSettings(
        LECTURE_OUTPUT_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/"),
        FRAME_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/frames/"),
        LOG_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/logs/"),
    )


@pytest.fixture
def setup_frames(cloud_settings):
    """Create fake frame images and return frames data."""
    lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
    frames_dir = Path(cloud_settings.frame_dir("lec_001"))
    lecture_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Create fake frame images.
    for i in range(1, 4):
        img_path = frames_dir / f"frame_{i:06d}.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # Minimal JPEG header.

    frames_data = [
        {"frame_id": 1, "timestamp": 5.0, "image_path": "frames/frame_000001.jpg"},
        {"frame_id": 2, "timestamp": 15.0, "image_path": "frames/frame_000002.jpg"},
        {"frame_id": 3, "timestamp": 25.0, "image_path": "frames/frame_000003.jpg"},
    ]
    return frames_data


class TestVisualUnderstanding:

    def test_run_visual_understanding_with_mock_model(self, cloud_settings, setup_frames):
        """VLM produces valid JSONL with mocked model."""
        from cloud.ingestion.qwen_pipeline.visual_understanding import (
            run_visual_understanding,
            _caption_single_frame,
        )

        # Create a mock model loader that returns validated VLMCaption.
        mock_model = MagicMock()
        mock_processor = MagicMock()

        call_count = 0

        def fake_caption(model, processor, image_path, frame_id):
            return VLMCaption(
                frame_id=frame_id,
                caption=f"Slide showing concept {frame_id}",
                objects=["diagram", "text", "formula"],
            )

        def mock_loader():
            return mock_model, mock_processor

        # Patch _caption_single_frame to use our fake.
        import cloud.ingestion.qwen_pipeline.visual_understanding as vu_module
        original_caption = vu_module._caption_single_frame
        vu_module._caption_single_frame = fake_caption

        try:
            count = run_visual_understanding(
                "lec_001", setup_frames,
                cloud_settings=cloud_settings,
                model_loader=mock_loader,
            )
        finally:
            vu_module._caption_single_frame = original_caption

        assert count == 3

        # Verify JSONL output.
        output_path = Path(cloud_settings.lecture_dir("lec_001")) / "vlm_output.jsonl"
        assert output_path.exists()

        records = []
        with open(output_path) as f:
            for line in f:
                records.append(json.loads(line))

        assert len(records) == 3
        assert records[0]["frame_id"] == 1
        assert "caption" in records[0]
        assert "objects" in records[0]

        # Validate each record against schema.
        for rec in records:
            VLMCaption(**rec)

    def test_vlm_caption_schema_rejects_empty_caption(self):
        """VLMCaption must reject empty captions."""
        with pytest.raises(Exception):
            VLMCaption(frame_id=1, caption="", objects=["test"])

    def test_vlm_output_is_jsonl_not_json_array(self, cloud_settings, setup_frames):
        """Output must be JSONL (one record per line), not a JSON array."""
        from cloud.ingestion.qwen_pipeline.visual_understanding import run_visual_understanding
        import cloud.ingestion.qwen_pipeline.visual_understanding as vu_module

        def fake_caption(model, processor, image_path, frame_id):
            return VLMCaption(
                frame_id=frame_id, caption=f"Caption {frame_id}", objects=["obj"],
            )

        original = vu_module._caption_single_frame
        vu_module._caption_single_frame = fake_caption

        try:
            run_visual_understanding(
                "lec_001", setup_frames,
                cloud_settings=cloud_settings,
                model_loader=lambda: (MagicMock(), MagicMock()),
            )
        finally:
            vu_module._caption_single_frame = original

        output_path = Path(cloud_settings.lecture_dir("lec_001")) / "vlm_output.jsonl"
        content = output_path.read_text()

        # JSONL: each line is a valid JSON object, not wrapped in [].
        assert not content.strip().startswith("[")
        lines = [l for l in content.strip().split("\n") if l.strip()]
        for line in lines:
            json.loads(line)  # Each line must parse independently.
