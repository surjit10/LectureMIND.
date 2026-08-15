# cloud/tests/test_json_repair.py
# Tests for the shared LLM JSON parsing pipeline used by A8 (entities),
# A9 (relations), and B1 (triplet queries).

import pytest

from cloud.utils.json_repair import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_REPAIRED,
    extract_json_array,
    parse_json_array,
    repair_invalid_escapes,
    strip_code_fences,
)


class TestParseJsonArray:

    def test_valid_array_ok(self):
        parsed, status = parse_json_array('[{"a": 1}, {"b": 2}]')
        assert status == STATUS_OK
        assert parsed == [{"a": 1}, {"b": 2}]

    def test_fenced_json_ok(self):
        parsed, status = parse_json_array('```json\n[{"a": 1}]\n```')
        assert status == STATUS_OK
        assert parsed == [{"a": 1}]

    def test_invalid_escapes_repaired(self):
        r"""W\_q style invalid escapes must be repaired, not dropped.

        The repair escapes the backslash so the JSON parses; the decoded
        content keeps the backslash ("content is never lost").
        """
        parsed, status = parse_json_array(r'[{"note": "W\_q"}]')
        assert status == STATUS_REPAIRED
        assert parsed[0]["note"] == r"W\_q"

    def test_unclosed_array_with_complete_objects_ok(self):
        """Missing closing `]` with a complete last object is recoverable."""
        parsed, status = parse_json_array('[{"a": 1}, {"b": 2}')
        assert status == STATUS_OK
        assert parsed == [{"a": 1}, {"b": 2}]

    def test_truncated_mid_object_salvages_prefix(self):
        """Truncation mid-object keeps the complete leading objects."""
        parsed, status = parse_json_array('[{"a": 1}, {"b": 2}, {"c":')
        assert status == STATUS_PARTIAL
        assert parsed == [{"a": 1}, {"b": 2}]

    def test_truncated_mid_string_salvages_prefix(self):
        """Truncation mid-string keeps objects emitted before the cut."""
        parsed, status = parse_json_array('[{"a": "x"}, {"b": "y')
        assert status == STATUS_PARTIAL
        assert parsed == [{"a": "x"}]

    def test_truncation_after_many_objects(self):
        """The measured A9 failure mode: long arrays truncated mid-object."""
        full = [{"source_entity_id": f"lec_entity_{i:06d}", "relation": "EXPLAINS",
                 "target_entity_id": f"lec_entity_{i + 1:06d}"} for i in range(1, 60)]
        raw = "[" + ",".join(__import__("json").dumps(o) for o in full[:40]) + ',{"so'
        parsed, status = parse_json_array(raw)
        assert status == STATUS_PARTIAL
        assert len(parsed) == 40
        assert parsed[0]["source_entity_id"] == "lec_entity_000001"

    def test_nested_objects_inside_values_survive(self):
        """Nested objects inside a value must not break the top-level scan."""
        raw = '[{"meta": {"nested": [1, 2]}}, {"b": 2'
        parsed, status = parse_json_array(raw)
        assert status == STATUS_PARTIAL
        assert parsed == [{"meta": {"nested": [1, 2]}}]

    def test_non_json_text_fails(self):
        parsed, status = parse_json_array("not json at all {}")
        assert status == STATUS_FAILED
        assert parsed is None

    def test_braces_inside_strings_ignored(self):
        """A literal `}` inside a string value must not close the object."""
        raw = '[{"text": "use } here"}, {"b": 2}, {"c":'
        parsed, status = parse_json_array(raw)
        assert status == STATUS_PARTIAL
        assert parsed == [{"text": "use } here"}, {"b": 2}]


class TestHelpers:

    def test_strip_code_fences(self):
        assert strip_code_fences("```json\n[1]\n```") == "[1]"
        assert strip_code_fences("[1]") == "[1]"

    def test_extract_json_array_skips_prefix_text(self):
        assert extract_json_array("sure, here: [1, 2]") == "[1, 2]"
        assert extract_json_array("no array here") is None

    def test_repair_invalid_escapes(self):
        assert repair_invalid_escapes(r'"W\_q"') == r'"W\\_q"'
        # Valid escapes must be untouched.
        assert repair_invalid_escapes(r'"a\tb"') == r'"a\tb"'
