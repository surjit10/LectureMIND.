"""Tests for Manifest schema."""
import pytest
from pydantic import ValidationError
from schemas.manifest import Manifest


VALID_MANIFEST = {
    "lecture_id": "lec_001",
    "created_at": "2026-06-17T12:00:00Z",
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "embedding_dimension": 1024,
    "chunk_count": 200,
    "segment_count": 10,
    "entity_count": 50,
    "relation_count": 30,
    "reranker_version": "v1.0",
}


def test_valid_manifest():
    m = Manifest(**VALID_MANIFEST)
    assert m.lecture_id == "lec_001"
    assert m.embedding_dimension == 1024
    assert m.entity_count == 50
    assert m.relation_count == 30


def test_embedding_dimension_must_be_1024():
    bad = {**VALID_MANIFEST, "embedding_dimension": 768}
    with pytest.raises(ValidationError, match="1024"):
        Manifest(**bad)


def test_embedding_dimension_zero_rejected():
    bad = {**VALID_MANIFEST, "embedding_dimension": 0}
    with pytest.raises(ValidationError):
        Manifest(**bad)


def test_empty_lecture_id_rejected():
    bad = {**VALID_MANIFEST, "lecture_id": ""}
    with pytest.raises(ValidationError):
        Manifest(**bad)


def test_empty_embedding_model_rejected():
    bad = {**VALID_MANIFEST, "embedding_model": "   "}
    with pytest.raises(ValidationError):
        Manifest(**bad)


def test_negative_chunk_count_rejected():
    bad = {**VALID_MANIFEST, "chunk_count": -1}
    with pytest.raises(ValidationError):
        Manifest(**bad)


def test_negative_entity_count_rejected():
    bad = {**VALID_MANIFEST, "entity_count": -5}
    with pytest.raises(ValidationError):
        Manifest(**bad)


def test_extra_field_rejected():
    bad = {**VALID_MANIFEST, "unexpected_field": "value"}
    with pytest.raises(ValidationError):
        Manifest(**bad)


def test_missing_required_field():
    bad = {k: v for k, v in VALID_MANIFEST.items() if k != "reranker_version"}
    with pytest.raises(ValidationError):
        Manifest(**bad)
