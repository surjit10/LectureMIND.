# cloud/tests/test_a2_transcriber.py
# Tests for Stage A2 — Speech Transcription.
# Mocks faster-whisper via model_loader injection — no downloads needed.

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from types import SimpleNamespace

from schemas.transcript import TranscriptSegment
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


@pytest.fixture
def mock_whisper_model():
    """Create a mock WhisperModel that returns realistic segments."""
    mock_model = MagicMock()
    fake_segments = [
        SimpleNamespace(start=10.5, end=15.2, text="BFS uses queue"),
        SimpleNamespace(start=15.3, end=20.0, text="DFS uses stack"),
        SimpleNamespace(start=20.1, end=25.5, text="Both are graph traversals"),
    ]
    fake_info = SimpleNamespace(language="en", language_probability=0.98)
    mock_model.transcribe.return_value = (iter(fake_segments), fake_info)
    return mock_model


class TestTranscriber:

    def test_transcribe_success(self, fake_video, cloud_settings, mock_whisper_model):
        """Valid transcription produces validated transcript.json."""
        from cloud.ingestion.whisper_pipeline.transcriber import transcribe

        result = transcribe(
            "lec_001", fake_video,
            cloud_settings=cloud_settings,
            model_loader=lambda: mock_whisper_model,
        )

        assert len(result) == 3
        assert all(isinstance(s, TranscriptSegment) for s in result)
        assert result[0].segment_id == 1
        assert result[0].start == 10.5
        assert result[0].text == "BFS uses queue"

        # Verify field names are exactly as spec requires.
        dump = result[0].model_dump()
        assert set(dump.keys()) == {"segment_id", "start", "end", "text"}

        # Verify output file.
        output_path = Path(cloud_settings.lecture_dir("lec_001")) / "transcript.json"
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert len(data) == 3
        assert data[0]["segment_id"] == 1

    def test_transcribe_missing_video(self, cloud_settings):
        """Missing video raises FileNotFoundError before model load."""
        from cloud.ingestion.whisper_pipeline.transcriber import transcribe

        with pytest.raises(FileNotFoundError):
            transcribe(
                "lec_001", Path("/nonexistent.mp4"),
                cloud_settings=cloud_settings,
                model_loader=lambda: MagicMock(),
            )

    def test_transcribe_empty_output_rejected(self, fake_video, cloud_settings):
        """Zero segments from Whisper must raise ValueError."""
        from cloud.ingestion.whisper_pipeline.transcriber import transcribe

        mock_model = MagicMock()
        fake_info = SimpleNamespace(language="en", language_probability=0.5)
        mock_model.transcribe.return_value = (iter([]), fake_info)

        with pytest.raises(ValueError, match="zero"):
            transcribe(
                "lec_001", fake_video,
                cloud_settings=cloud_settings,
                model_loader=lambda: mock_model,
            )
