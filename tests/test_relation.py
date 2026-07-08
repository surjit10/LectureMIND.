"""Tests for Relation schema and RelationType enum."""
import pytest
from pydantic import ValidationError
from schemas.relation import Relation
from schemas.enums import RelationType


VALID_RELATION = {
    "relation_id": "rel_001",
    "source_entity_id": "lec_001_entity_000012",
    "relation": RelationType.PREREQUISITE_OF,
    "target_entity_id": "lec_001_entity_000037",
}


def test_valid_relation():
    r = Relation(**VALID_RELATION)
    assert r.relation_id == "rel_001"
    assert r.source_entity_id == "lec_001_entity_000012"
    assert r.relation == RelationType.PREREQUISITE_OF
    assert r.target_entity_id == "lec_001_entity_000037"


def test_all_relation_types_valid():
    for rt in RelationType:
        r = Relation(
            relation_id="rel_002",
            source_entity_id="lec_001_entity_000001",
            relation=rt,
            target_entity_id="lec_001_entity_000002",
        )
        assert r.relation == rt


def test_invalid_relation_type_rejected():
    bad = {**VALID_RELATION, "relation": "INVENTED_RELATION"}
    with pytest.raises(ValidationError):
        Relation(**bad)


def test_empty_source_entity_id_rejected():
    bad = {**VALID_RELATION, "source_entity_id": ""}
    with pytest.raises(ValidationError):
        Relation(**bad)


def test_empty_target_entity_id_rejected():
    bad = {**VALID_RELATION, "target_entity_id": "  "}
    with pytest.raises(ValidationError):
        Relation(**bad)


def test_empty_relation_id_rejected():
    bad = {**VALID_RELATION, "relation_id": ""}
    with pytest.raises(ValidationError):
        Relation(**bad)


def test_no_source_name_field():
    """Relations must use entity IDs, not names."""
    bad = {**VALID_RELATION, "source": "Queue"}
    with pytest.raises(ValidationError):
        Relation(**bad)


def test_relation_type_enum_is_closed():
    """Verify enum has exactly the 6 specified relation types."""
    expected = {
        "PREREQUISITE_OF",
        "INTRODUCED_BEFORE",
        "USED_BY",
        "DERIVED_FROM",
        "VISUALIZED_BY",
        "EXPLAINS",
    }
    actual = {r.value for r in RelationType}
    assert actual == expected
