# local/llm/ollama_backend.py
# OllamaBackend — local offline LLM provider.
#
# Wraps the existing ollama SDK call that was previously embedded in
# answer_generator.py and learning_service.py.
# Logic is identical — simply moved behind the LLMBackend interface.

import logging
from typing import Any, Dict, List, Optional, Tuple

from local.llm.base import LLMBackend, LLMGenerationResult

logger = logging.getLogger(__name__)


class OllamaBackend(LLMBackend):
    """
    Ollama local-model backend.

    Communicates with a running Ollama daemon at localhost:11434 (default).
    Supports an injected client for testing (avoids real Ollama dependency).
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        ollama_client: Optional[Any] = None,
    ):
        self._model = model
        self._client = ollama_client  # injected in tests; None → real ollama

    @property
    def provider_name(self) -> str:
        return "Ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import ollama
        return ollama

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Backward compatible generate method."""
        return self.generate_with_metadata(messages).answer

    def generate_with_metadata(self, messages: List[Dict[str, str]]) -> LLMGenerationResult:
        """Call ollama.chat() and return the assistant reply with metadata."""
        try:
            client = self._get_client()
            response = client.chat(model=self._model, messages=messages)

            metadata = {"model": self._model, "provider": self.provider_name}
            if hasattr(response, "prompt_eval_count"):
                metadata["prompt_tokens"] = response.prompt_eval_count
                metadata["completion_tokens"] = getattr(response, "eval_count", 0)
            elif isinstance(response, dict):
                metadata["prompt_tokens"] = response.get("prompt_eval_count", 0)
                metadata["completion_tokens"] = response.get("eval_count", 0)

            if hasattr(response, "message"):
                return LLMGenerationResult(answer=response.message.content.strip(), metadata=metadata)
            elif isinstance(response, dict):
                return LLMGenerationResult(answer=response.get("message", {}).get("content", "").strip(), metadata=metadata)
            return LLMGenerationResult(answer=str(response).strip(), metadata=metadata)

        except Exception as exc:
            logger.error("OllamaBackend: generation failed: %s", exc)
            return LLMGenerationResult(answer="", metadata={"error": str(exc)})

    def health_check(self) -> bool:
        """Return True if Ollama is running and the model is available."""
        try:
            client = self._get_client()
            models_response = client.list()

            available: set = set()
            if hasattr(models_response, "models"):
                for m in models_response.models:
                    name = getattr(m, "name", "")
                    available.add(name)
                    available.add(name.split(":")[0])
            elif isinstance(models_response, dict) and "models" in models_response:
                for m in models_response["models"]:
                    name = m.get("name", "")
                    available.add(name)
                    available.add(name.split(":")[0])

            ready = self._model in available or self._model.split(":")[0] in available
            if ready:
                logger.info("OllamaBackend: model '%s' is ready.", self._model)
            else:
                logger.warning(
                    "OllamaBackend: model '%s' not found. Available: %s",
                    self._model, available,
                )
            return ready

        except Exception as exc:
            logger.error("OllamaBackend: health check failed: %s", exc)
            return False

    def list_local_models(self) -> list:
        """Return all locally available Ollama model names."""
        try:
            client = self._get_client()
            models_response = client.list()
            names = []
            if hasattr(models_response, "models"):
                for m in models_response.models:
                    names.append(getattr(m, "name", ""))
            elif isinstance(models_response, dict) and "models" in models_response:
                for m in models_response["models"]:
                    names.append(m.get("name", ""))
            return [n for n in names if n]
        except Exception as exc:
            logger.error("OllamaBackend: list models failed: %s", exc)
            return []
