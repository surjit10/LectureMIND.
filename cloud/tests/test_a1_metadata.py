# cloud/tests/test_a1_metadata.py
# Tests for Stage A1 — Video Metadata Extraction.
# Uses OpenCV mock — no real video files needed.

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from schemas.metadata import VideoMetadata
from config import CloudSettings


@pytest.fixture
def cloud_settings(tmp_path):
    """CloudSettings pointing to a temporary directory."""
    return CloudSettings(
        LECTURE_OUTPUT_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/"),
        FRAME_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/frames/"),
        LOG_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/logs/"),
    )


@pytest.fixture
def fake_video(tmp_path):
    """Create a dummy video file."""
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"fake_video_data")
    return video_path


class TestMetadataExtractor:
    """Tests for extract_metadata."""

    @patch("cloud.ingestion.metadata.metadata_extractor.cv2")
    def test_extract_metadata_success(self, mock_cv2, fake_video, cloud_settings):
        """Valid video produces valid metadata.json."""
        from cloud.ingestion.metadata.metadata_extractor import extract_metadata

        # Mock VideoCapture.
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            5: 30.0,  # CAP_PROP_FPS
            7: 216000.0,  # CAP_PROP_FRAME_COUNT (7200s * 30fps)
        }.get(prop, 0.0)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_COUNT = 7

        result = extract_metadata("lec_001", fake_video, cloud_settings=cloud_settings)

        assert isinstance(result, VideoMetadata)
        assert result.lecture_id == "lec_001"
        assert result.duration == 7200.0
        assert result.fps == 30.0

        # Verify output file written.
        output_path = Path(cloud_settings.lecture_dir("lec_001")) / "metadata.json"
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["lecture_id"] == "lec_001"
        assert data["duration"] == 7200.0

    def test_extract_metadata_missing_video(self, cloud_settings):
        """Missing video file raises FileNotFoundError."""
        from cloud.ingestion.metadata.metadata_extractor import extract_metadata

        with pytest.raises(FileNotFoundError):
            extract_metadata("lec_001", Path("/nonexistent.mp4"), cloud_settings=cloud_settings)

    @patch("cloud.ingestion.metadata.metadata_extractor.cv2")
    def test_extract_metadata_unopenable_video(self, mock_cv2, fake_video, cloud_settings):
        """Unopenable video raises ValueError."""
        from cloud.ingestion.metadata.metadata_extractor import extract_metadata

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        with pytest.raises(ValueError, match="Cannot open"):
            extract_metadata("lec_001", fake_video, cloud_settings=cloud_settings)
