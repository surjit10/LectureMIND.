import pytest

from schemas.prerequisite import validate_prerequisite_artifact


def test_valid_prerequisite_dag():
    edges = validate_prerequisite_artifact(
        {"prerequisites": [
            {"source_id": "a", "target_id": "b", "confidence": 0.8},
            {"source_id": "b", "target_id": "c", "confidence": 0.7},
        ]},
        {"a", "b", "c"},
    )
    assert len(edges) == 2


def test_prerequisite_cycle_is_rejected():
    with pytest.raises(ValueError, match="directed cycle"):
        validate_prerequisite_artifact(
            {"prerequisites": [
                {"source_id": "a", "target_id": "b", "confidence": 0.8},
                {"source_id": "b", "target_id": "a", "confidence": 0.7},
            ]},
            {"a", "b"},
        )
