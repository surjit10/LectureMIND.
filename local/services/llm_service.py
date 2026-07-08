# local/services/llm_service.py
# LLM readiness service wrapper for Ollama.
#
# Verifies model availability. Does NOT generate answers or prompts.
# That is deferred to Chunk 5 (DSPy/LangGraph integration).
#
# Environment: Local only.

import logging
from typing import Any, Optional

from config import LocalSettings
from local.loaders.ollama_loader import check_model_ready

logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM service wrapper — readiness checks only.

    Usage:
        service = LLMService()
        if service.is_ready():
            ...  # Chunk 5 will add generation logic.
    """

    def __init__(
        self,
        local_settings: LocalSettings | None = None,
        ollama_client: Optional[Any] = None,
    ):
        self._settings = local_settings or LocalSettings()
        self._ollama_client = ollama_client

    @property
    def model_name(self) -> str:
        return self._settings.OLLAMA_MODEL

    def is_ready(self) -> bool:
        """Check if the Ollama model is available and ready."""
        return check_model_ready(
            local_settings=self._settings,
            ollama_client=self._ollama_client,
        )
