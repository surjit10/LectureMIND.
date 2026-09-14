"""Shared lemma/alias-aware name matching for KG auditors.

Problem this solves: exact string comparison fails on morphological variants
(``transformers`` vs ``Transformer``, ``coil`` vs ``Coils``), which inflates
false negatives in every gold-reference metric. The auditors previously used
plain ``lower().strip()`` comparison; this module centralizes a conservative
normalization so all three auditors agree on what "the same concept name"
means.

Design constraints:
- Conservative: only deterministic, language-agnostic rules. No stemming
  library dependency, no aggressive fuzzy matching — a match must be
  defensible in a review.
- Alias-aware: gold files may carry ``aliases`` lists; they are folded into
  the matcher by the callers via :class:`GoldNameIndex`.
- Transparency: every matched pair is classified by the rule that matched it
  (``EXACT`` / ``ALIAS`` / ``LEMMA``), so audits can report the matching
  method instead of silently broadening the definition of a match.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Deterministic singular/plural normalization for regular English nouns.
# Ordered longest-first so e.g. "entities" is handled before "s".
_PLURAL_RULES: List[Tuple[str, str]] = [
    ("ies", "y"),      # entities -> entity, liabilities -> liability
    ("ves", "f"),      # leaves -> leaf (approximate; conservative set)
    ("xes", "x"),      # boxes -> box
    ("ches", "ch"),    # branches -> branch
    ("shes", "sh"),    # dashes -> dash
    ("sses", "ss"),    # classes -> class
    ("ses", "s"),      # uses -> use (after the -sses rule)
    ("zes", "z"),      # sizes -> size
    ("s", ""),         # generic final-s
]

_TOKEN_SPLIT = re.compile(r"[\s\-_]+")


def normalize_name(name: Optional[str]) -> str:
    """Lowercase, strip punctuation/underscores, collapse whitespace."""
    s = (name or "").strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _singularize_token(token: str) -> str:
    if len(token) <= 3:
        return token
    for suffix, replacement in _PLURAL_RULES:
        if token.endswith(suffix):
            candidate = token[: -len(suffix)] + replacement
            # Guard against degenerate results ("as" -> "a", "is" -> "i")
            if len(candidate) >= 3:
                return candidate
            return token
    return token


def lemma_key(name: Optional[str]) -> str:
    """Morphology-insensitive key: token order preserved, plurals folded.

    ``transformers`` and ``transformer`` share a key; ``transformer`` and
    ``rectifier`` do not. Multi-word names must match token-by-token, so
    ``direct currents`` matches ``direct current`` but not ``current``.
    """
    tokens = [t for t in _TOKEN_SPLIT.split(normalize_name(name)) if t]
    return " ".join(_singularize_token(t) for t in tokens)


def names_match(predicted: Optional[str], gold: Optional[str]) -> Tuple[bool, str]:
    """Return ``(matched, match_type)`` for a predicted/gold name pair.

    Match types: ``EXACT`` (normalized equality), ``LEMMA`` (plurality-insensitive
    equality). Alias handling lives in :class:`GoldNameIndex`.
    """
    p, g = normalize_name(predicted), normalize_name(gold)
    if not p or not g:
        return False, ""
    if p == g:
        return True, "EXACT"
    if lemma_key(p) == lemma_key(g):
        return True, "LEMMA"
    return False, ""


class GoldNameIndex:
    """Index of gold canonical names + aliases supporting one-to-one lookup.

    Each gold concept may be claimed at most once; the index enforces that
    via ``claim``. Lookup order is EXACT -> ALIAS -> LEMMA over canonical
    names and aliases, so the strictest match wins.
    """

    def __init__(self, concepts: Iterable[Dict]) -> None:
        # canonical_key -> {"display": canonical_name, "claims": 0}
        self._by_canonical: Dict[str, Dict] = {}
        # alias_key -> set of canonical_keys it can resolve to
        self._alias_to_canonical: Dict[str, Set[str]] = {}
        for concept in concepts:
            cname = concept.get("canonical_name") or concept.get("name") or ""
            if not cname:
                continue
            key = normalize_name(cname)
            if key not in self._by_canonical:
                self._by_canonical[key] = {"display": cname, "claims": 0}
            for alias in concept.get("aliases", []) or []:
                akey = normalize_name(alias)
                if akey:
                    self._alias_to_canonical.setdefault(akey, set()).add(key)

    def match(self, predicted_name: Optional[str]) -> Optional[Tuple[str, str]]:
        """Find an unclaimed gold concept matching ``predicted_name``.

        Returns ``(canonical_key, match_type)`` or ``None``.
        """
        p = normalize_name(predicted_name)
        if not p:
            return None

        # 1. Exact canonical
        if p in self._by_canonical and self._by_canonical[p]["claims"] == 0:
            return p, "EXACT"
        # 2. Exact alias
        if p in self._alias_to_canonical:
            for ckey in sorted(self._alias_to_canonical[p]):
                if self._by_canonical[ckey]["claims"] == 0:
                    return ckey, "ALIAS"
        # 3. Lemma match over canonicals
        plem = lemma_key(p)
        for ckey, info in self._by_canonical.items():
            if info["claims"] == 0 and lemma_key(ckey) == plem:
                return ckey, "LEMMA"
        # 4. Lemma match over aliases
        for akey, ckeys in self._alias_to_canonical.items():
            if lemma_key(akey) == plem:
                for ckey in sorted(ckeys):
                    if self._by_canonical[ckey]["claims"] == 0:
                        return ckey, "ALIAS_LEMMA"
        return None

    def claim(self, canonical_key: str) -> None:
        self._by_canonical[canonical_key]["claims"] += 1

    def claimed_count(self) -> int:
        return sum(1 for info in self._by_canonical.values() if info["claims"] > 0)

    def size(self) -> int:
        return len(self._by_canonical)

    def display_name(self, canonical_key: str) -> str:
        return self._by_canonical[canonical_key]["display"]
