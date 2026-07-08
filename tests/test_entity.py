"""Tests for Entity schema and EntityType enum."""
import pytest
from pydantic import ValidationError
from schemas.entity import Entity
from schemas.enums import EntityType


VALID_ENTITY = {
    "entity_id": "lec_001_entity_000123",
    "name": "BFS",
    "type": EntityType.Algorithm,
}


def test_valid_entity():
    e = Entity(**VALID_ENTITY)
    assert e.entity_id == "lec_001_entity_000123"
    assert e.type == EntityType.Algorithm


def test_all_entity_types_valid():
    for et in EntityType:
        e = Entity(entity_id="lec_001_entity_000001", name="test", type=et)
        assert e.type == et


def test_invalid_entity_type_rejected():
    bad = {**VALID_ENTITY, "type": "InvalidType"}
    with pytest.raises(ValidationError):
        Entity(**bad)


def test_empty_entity_id_rejected():
    bad = {**VALID_ENTITY, "entity_id": ""}
    with pytest.raises(ValidationError):
        Entity(**bad)


def test_empty_name_rejected():
    bad = {**VALID_ENTITY, "name": "  "}
    with pytest.raises(ValidationError):
        Entity(**bad)


def test_extra_field_rejected():
    bad = {**VALID_ENTITY, "lecture_id": "lec_001"}
    with pytest.raises(ValidationError):
        Entity(**bad)


def test_entity_type_enum_is_closed():
    """Verify enum has exactly the 6 specified types."""
    expected = {"Concept", "Algorithm", "Formula", "Code", "Diagram", "LectureSegment"}
    actual = {e.value for e in EntityType}
    assert actual == expected
