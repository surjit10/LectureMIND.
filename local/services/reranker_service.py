# local/services/reranker_service.py
# Reranker scoring service wrapper.
#
# Provides score_pairs(query, passages) → [(passage, score)] sorted descending.
# No retrieval logic — only scoring.
#
# Environment: Local only.

import logging
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Wrapper around a loaded CrossEncoder for passage scoring.

    Usage:
        service = RerankerService(model)
        ranked = service.score_pairs("query", ["passage1", "passage2"])
    """

    def __init__(self, model: Any):
        """
        Args:
            model: A loaded CrossEncoder (or compatible predict() interface).
        """
        self._model = model

    def score_pairs(
        self,
        query: str,
        passages: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Score query-passage pairs and return sorted by score descending.

        Args:
            query: The user query string.
            passages: List of candidate passage strings.

        Returns:
            List of (passage, score) tuples sorted by score descending.
        """
        if not passages:
            return []

        pairs = [(query, passage) for passage in passages]
        scores = self._model.predict(pairs)

        # Combine passages with scores and sort descending.
        scored = list(zip(passages, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            "RerankerService: Scored %d passages for query '%s...'",
            len(passages), query[:50],
        )
        return scored
