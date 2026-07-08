# cloud/tests/test_a5_ocr.py
# Tests for Stage A5 — PaddleOCR.
# OCR engine is mocked — no downloads.

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from schemas.ocr import OCRResult
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

    for i in range(1, 4):
        img_path = frames_dir / f"frame_{i:06d}.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    return [
        {"frame_id": 1, "timestamp": 5.0, "image_path": "frames/frame_000001.jpg"},
        {"frame_id": 2, "timestamp": 15.0, "image_path": "frames/frame_000002.jpg"},
        {"frame_id": 3, "timestamp": 25.0, "image_path": "frames/frame_000003.jpg"},
    ]


class TestOCREngine:

    def test_run_ocr_with_mock_engine(self, cloud_settings, setup_frames):
        """OCR produces valid JSONL with mocked engine."""
        from cloud.ingestion.paddleocr_pipeline.ocr_engine import run_ocr

        # Mock OCR engine that returns realistic results.
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [
            [
                [[[0, 0], [100, 0], [100, 30], [0, 30]], ("Breadth First Search", 0.95)],
                [[[0, 40], [100, 40], [100, 70], [0, 70]], ("O(V+E)", 0.88)],
            ]
        ]

        count = run_ocr("lec_001", setup_frames, cloud_settings=cloud_settings, ocr_engine=mock_engine)

        assert count == 3

        # Verify JSONL output.
        output_path = Path(cloud_settings.lecture_dir("lec_001")) / "ocr_output.jsonl"
        assert output_path.exists()

        records = []
        with open(output_path) as f:
            for line in f:
                records.append(json.loads(line))

        assert len(records) == 3
        assert records[0]["frame_id"] == 1
        assert "Breadth First Search" in records[0]["ocr_text"]
        assert "O(V+E)" in records[0]["ocr_text"]

        # Validate each record.
        for rec in records:
            OCRResult(**rec)

    def test_ocr_empty_result(self, cloud_settings, setup_frames):
        """OCR with no text detected produces empty ocr_text list."""
        from cloud.ingestion.paddleocr_pipeline.ocr_engine import run_ocr

        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [None]

        count = run_ocr("lec_001", setup_frames, cloud_settings=cloud_settings, ocr_engine=mock_engine)
        assert count == 3

        output_path = Path(cloud_settings.lecture_dir("lec_001")) / "ocr_output.jsonl"
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                assert rec["ocr_text"] == []

    def test_ocr_output_is_jsonl_format(self, cloud_settings, setup_frames):
        """Output must be JSONL, not JSON array."""
        from cloud.ingestion.paddleocr_pipeline.ocr_engine import run_ocr

        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [[]]

        run_ocr("lec_001", setup_frames, cloud_settings=cloud_settings, ocr_engine=mock_engine)

        output_path = Path(cloud_settings.lecture_dir("lec_001")) / "ocr_output.jsonl"
        content = output_path.read_text()
        assert not content.strip().startswith("[")
