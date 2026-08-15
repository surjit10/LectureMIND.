"""Tests for MultimodalChunk — the master contract."""
import pytest
from pydantic import ValidationError
from schemas.chunk import MultimodalChunk


VALID_CHUNK = {
    "lecture_id": "lec_001",
    "chunk_id": "lec_001_chunk_000001",
    "timestamp": 20.5,
    "transcript": "BFS uses queue",
    "visual_context": "BFS traversal diagram",
    "ocr_text": "Breadth First Search O(V+E)",
    "start_time": 10.0,
    "end_time": 15.0,
    "segment_ids": [1, 2, 3],
}


def test_valid_chunk():
    c = MultimodalChunk(**VALID_CHUNK)
    assert c.lecture_id == "lec_001"
    assert c.chunk_id == "lec_001_chunk_000001"
    assert c.timestamp == 20.5


def test_empty_lecture_id_rejected():
    bad = {**VALID_CHUNK, "lecture_id": ""}
    with pytest.raises(ValidationError):
        MultimodalChunk(**bad)


def test_empty_chunk_id_rejected():
    bad = {**VALID_CHUNK, "chunk_id": "   "}
    with pytest.raises(ValidationError):
        MultimodalChunk(**bad)


def test_negative_timestamp_rejected():
    bad = {**VALID_CHUNK, "timestamp": -1.0}
    with pytest.raises(ValidationError):
        MultimodalChunk(**bad)


def test_zero_timestamp_valid():
    c = MultimodalChunk(**{**VALID_CHUNK, "timestamp": 0.0})
    assert c.timestamp == 0.0


def test_extra_field_rejected():
    bad = {**VALID_CHUNK, "segment_id": "seg_5"}
    with pytest.raises(ValidationError):
        MultimodalChunk(**bad)


def test_chunk_has_no_segment_id():
    """segment_id is intentionally absent from MultimodalChunk — assigned by A7."""
    c = MultimodalChunk(**VALID_CHUNK)
    assert not hasattr(c, "segment_id")


def test_provenance_fields_default_from_legacy_chunk():
    """A legacy 6-field chunk (old packages) gets default provenance values."""
    legacy = {k: v for k, v in VALID_CHUNK.items() if k in {
        "lecture_id", "chunk_id", "timestamp", "transcript",
        "visual_context", "ocr_text",
    }}
    c = MultimodalChunk(**legacy)
    assert c.start_time == 0.0
    assert c.end_time == 0.0
    assert c.segment_ids == []


def test_provenance_fields_set_correctly():
    """Provenance fields carry merged-chunk metadata."""
    c = MultimodalChunk(**VALID_CHUNK)
    assert c.start_time == 10.0
    assert c.end_time == 15.0
    assert c.segment_ids == [1, 2, 3]


def test_negative_end_time_rejected():
    bad = {**VALID_CHUNK, "end_time": -1.0}
    with pytest.raises(ValidationError):
        MultimodalChunk(**bad)


def test_negative_start_time_rejected():
    bad = {**VALID_CHUNK, "start_time": -5.0}
    with pytest.raises(ValidationError):
        MultimodalChunk(**bad)
