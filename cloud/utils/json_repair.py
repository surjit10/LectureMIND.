# cloud/utils/json_repair.py
# Centralized LLM JSON parsing for Stages A8 (entities), A9 (relations), and
# B1 (triplet queries).
#
# Previously each stage implemented its own fence-stripping / array-extraction /
# escape-repair logic, and A9 was missing the escape repair entirely — so a
# response containing e.g. "W\_q" or "\(" raised "JSON parse error: Invalid
# \escape" and the whole segment's output was discarded. This module
# centralizes the shared pipeline so a fix lands in exactly one place:
#
#   raw LLM response
#     -> strip markdown code fences safely
#     -> extract the first balanced top-level JSON array
#     -> strict json.loads()                       -> status "ok"
#     -> deterministic repair of invalid escapes   -> status "repaired"
#     -> json.loads() again
#     -> salvage complete top-level objects from a truncated array
#                                                  -> status "partial"
#     -> if still nothing parseable: (None, "failed") — the caller decides.
#
# SAFETY: repairs are minimal and deterministic. We only escape a backslash
# that precedes a character JSON forbids inside a string literal (anything
# other than \ " / b f n r t u). Valid escapes, unicode escapes (\\u....), and
# LaTeX commands such as \\times are NOT rewritten (\\t is a valid JSON tab
# escape — mangled, but the content is never lost). No semantic content is
# fabricated; salvage keeps only objects that are complete and parseable as
# emitted by the model. If repair cannot succeed we report "failed".

import json
import logging
import re
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# A backslash not followed by one of the JSON-legal escape letters is invalid.
_INVALID_ESCAPE = re.compile(r'\\(?![\\"/bfnrtu])')

_FENCE_OPEN = re.compile(r"^```[a-zA-Z]*\s*")
_FENCE_CLOSE = re.compile(r"```\s*$")

# Parse statuses.
STATUS_OK = "ok"
STATUS_REPAIRED = "repaired"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"


def strip_code_fences(text: str) -> str:
    """Strip surrounding markdown code fences (```json ... ```) safely."""
    t = text.strip()
    if t.startswith("```"):
        t = _FENCE_OPEN.sub("", t)
        t = _FENCE_CLOSE.sub("", t)
    return t.strip()


def extract_json_array(text: str) -> Optional[str]:
    """Return the first top-level balanced JSON array, or None.

    The scan is string-aware so a '[' inside a string value is never treated
    as a bracket.
    """
    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def repair_invalid_escapes(json_str: str) -> str:
    """Escape backslashes that precede characters JSON forbids in strings.

    Fixes the measured failure mode ("JSON parse error: Invalid \\escape") for
    outputs such as W\\_q, \\(, \\), \\[, \\] without touching valid escapes.
    """
    return _INVALID_ESCAPE.sub(r"\\\\", json_str)


def _try_load(candidate: str) -> Optional[Any]:
    """json.loads with the safe-escape repair as a second attempt."""
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            return json.loads(repair_invalid_escapes(candidate))
        except json.JSONDecodeError:
            return None


def _salvage_complete_objects(text: str) -> Optional[List[Any]]:
    """Salvage complete top-level JSON objects from truncated text.

    The measured failure mode is max_tokens truncation mid-array: the closing
    `]` and the tail objects are missing, so the whole segment used to be
    dropped. This scan walks the text string-aware (a `{` or `}` inside a
    string value is never treated as a brace), collects every top-level
    `{...}` span that is balanced and parseable, and returns them in order.

    Returns:
        The salvaged objects, or None when no complete object survives.
    """
    start = text.find("[")
    if start == -1:
        return None

    objects: List[Any] = []
    i = start
    n = len(text)
    in_string = False
    escape = False
    brace_depth = 0
    obj_start = -1

    while i < n:
        ch = text[i]
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string and ch == "{":
            if brace_depth == 0:
                obj_start = i
            brace_depth += 1
        elif not in_string and ch == "}":
            if brace_depth > 0:
                brace_depth -= 1
                if brace_depth == 0 and obj_start != -1:
                    parsed = _try_load(text[obj_start : i + 1])
                    if parsed is not None:
                        objects.append(parsed)
                    obj_start = -1
        i += 1

    return objects if objects else None


def parse_json_array(raw_text: str) -> Tuple[Optional[List[Any]], str]:
    """Parse an LLM response into a JSON array.

    Pipeline: strip fences -> extract balanced array -> strict load -> safe
    escape repair -> load again. Bounded recovery is attempted for unclosed
    arrays (missing closing `]` or unterminated trailing string). When the
    response was truncated mid-array, the complete leading objects are
    salvaged instead of dropping the whole response.

    Returns:
        (parsed, status) where status is one of STATUS_OK / STATUS_REPAIRED /
        STATUS_PARTIAL / STATUS_FAILED. STATUS_PARTIAL means only a prefix of
        the objects survived truncation. `parsed` is None when status is
        STATUS_FAILED.
    """
    text = strip_code_fences(raw_text)

    json_str = extract_json_array(text)
    candidates: List[str] = []
    if json_str is not None:
        candidates.append(json_str)
    else:
        # Unclosed array — recover the rest of the response, optionally
        # closing with `]` or `"]` (the common truncation modes).
        start = text.find("[")
        if start != -1:
            rest = text[start:]
            candidates.append(rest + "]")
            candidates.append(rest + '"]')

    for candidate in candidates:
        # Strict load first: only a response that parses as emitted is "ok".
        try:
            return json.loads(candidate), STATUS_OK
        except json.JSONDecodeError:
            pass
        # Escape repair second: a response that needed repair is "repaired".
        repaired = repair_invalid_escapes(candidate)
        parsed = _try_load(repaired)
        if parsed is not None:
            return parsed, STATUS_REPAIRED

    # Truncated mid-array: keep the complete objects that were emitted before
    # the cut instead of dropping the entire response.
    salvaged = _salvage_complete_objects(text)
    if salvaged:
        return salvaged, STATUS_PARTIAL

    return None, STATUS_FAILED
