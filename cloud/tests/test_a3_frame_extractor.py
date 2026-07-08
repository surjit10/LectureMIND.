# cloud/tests/test_a3_frame_extractor.py
# Tests for Stage A3 — Frame Extraction (SSIM slide-change detection).
# Uses synthetic frames via OpenCV mock.

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

from schemas.frames import Frame
from config import CloudSettings


@pytest.fixture
def cloud_settings(tmp_path):
    return CloudSettings(
        LECTURE_OUTPUT_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/"),
        FRAME_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/frames/"),
        LOG_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/logs/"),
    )


@pytest.fixture
def fake_video(tmp_path):
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"fake_video_data")
    return video_path


class TestFrameExtractor:

    def test_ssim_identical_images(self):
        """SSIM of identical images should be ~1.0."""
        from cloud.ingestion.frame_extraction.frame_extractor import _compute_ssim

        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        ssim = _compute_ssim(img, img)
        assert ssim > 0.99

    def test_ssim_different_images(self):
        """SSIM of very different images should be low."""
        from cloud.ingestion.frame_extraction.frame_extractor import _compute_ssim

        img1 = np.zeros((100, 100), dtype=np.uint8)
        img2 = np.full((100, 100), 255, dtype=np.uint8)
        ssim = _compute_ssim(img1, img2)
        assert ssim < 0.1

    @patch("cloud.ingestion.frame_extraction.frame_extractor.cv2")
    def test_extract_frames_produces_key_frames(self, mock_cv2, fake_video, cloud_settings):
        """Frame extraction with slide changes produces fewer key frames."""
        from cloud.ingestion.frame_extraction.frame_extractor import extract_frames

        # Create synthetic frames — 3 identical then 1 different (slide change).
        frame_white = np.full((480, 640, 3), 200, dtype=np.uint8)
        frame_dark = np.full((480, 640, 3), 30, dtype=np.uint8)

        frames_sequence = [frame_white, frame_white, frame_white, frame_dark]
        frame_iter = iter([(True, f) for f in frames_sequence] + [(False, None)])

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # FPS=1.0 so frame_step=1 and all frames are evaluated.
        mock_cap.get.side_effect = lambda prop: {5: 1.0}.get(prop, 0.0)
        mock_cap.read.side_effect = lambda: next(frame_iter)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.cvtColor.side_effect = lambda img, code: img[:, :, 0]
        mock_cv2.COLOR_BGR2GRAY = 6
        mock_cv2.resize.side_effect = lambda img, size: img[:size[1], :size[0]]
        mock_cv2.imwrite.return_value = True

        # cooldown=0 so the test isn't affected by cooldown logic.
        result = extract_frames("lec_001", fake_video, cloud_settings=cloud_settings,
                                cooldown_seconds=0.0)

        # Should extract 2 key frames: first frame + the slide change.
        assert len(result) == 2
        assert all(isinstance(f, Frame) for f in result)
        assert result[0].frame_id == 1
        assert result[1].frame_id == 2

        # Verify output file.
        output_path = Path(cloud_settings.lecture_dir("lec_001")) / "frames.json"
        assert output_path.exists()

    def test_extract_frames_missing_video(self, cloud_settings):
        """Missing video raises FileNotFoundError."""
        from cloud.ingestion.frame_extraction.frame_extractor import extract_frames

        with pytest.raises(FileNotFoundError):
            extract_frames("lec_001", Path("/nonexistent.mp4"), cloud_settings=cloud_settings)

    def test_frame_schema_validation(self):
        """Frame schema must reject extra fields like segment_id."""
        with pytest.raises(Exception):
            Frame(frame_id=1, timestamp=1.0, image_path="frames/f.jpg", segment_id="seg_1")
