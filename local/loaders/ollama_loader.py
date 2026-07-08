# local/loaders/ollama_loader.py
# D4 — Ollama Model Verification.
#
# Verifies that Ollama is running and the required model
# (llama3.1:8b-instruct-q4_K_M) is available.
#
# Does NOT generate answers, build prompts, or create agent logic.
# Only verifies readiness.
#
# Environment: Local only. Never imports from cloud/.

import logging
from typing import Any, Dict, Optional

from config import LocalSettings

logger = logging.getLogger(__name__)


class OllamaLoadError(Exception):
    """Raised when Ollama verification fails."""
    pass


def check_model_ready(
    local_settings: LocalSettings | None = None,
    ollama_client: Optional[Any] = None,
    model_name: Optional[str] = None,
) -> bool:
    """
    Verify that Ollama is running and the required model is available.

    Args:
        local_settings: Injected LocalSettings.
        ollama_client: Optional pre-created Ollama client for testing.
        model_name: Optional explicit model name to check.

    Returns:
        True if model is ready, False otherwise.
    """
    settings = local_settings or LocalSettings()
    model_to_check = model_name or settings.OLLAMA_MODEL

    try:
        if ollama_client is not None:
            client = ollama_client
        else:
            import ollama
            client = ollama

        # List available models.
        models_response = client.list()

        # Extract model names from the response.
        available = set()
        if hasattr(models_response, "models"):
            # ollama SDK response object.
            for m in models_response.models:
                # The newer SDK uses 'model', older might use 'name'
                name = getattr(m, "model", getattr(m, "name", ""))
                if name:
                    available.add(name.split(":")[0])
                    available.add(name)
        elif isinstance(models_response, dict) and "models" in models_response:
            # Dict response.
            for m in models_response["models"]:
                name = m.get("model", m.get("name", ""))
                if name:
                    available.add(name.split(":")[0])
                    available.add(name)

        if model_to_check in available or model_to_check.split(":")[0] in available:
            logger.info("D4: Ollama model '%s' is ready.", model_to_check)
            return True
        else:
            logger.warning(
                "D4: Model '%s' not found. Available: %s",
                model_to_check, available,
            )
            return False

    except Exception as exc:
        logger.error("D4: Cannot connect to Ollama: %s", exc)
        return False


def pull_model_if_needed(
    local_settings: LocalSettings | None = None,
    ollama_client: Optional[Any] = None,
    model_name: Optional[str] = None,
) -> bool:
    """
    Pull the required model if not already available.

    Returns:
        True if model is ready after pull attempt.
    """
    settings = local_settings or LocalSettings()
    model_to_pull = model_name or settings.OLLAMA_MODEL

    if check_model_ready(local_settings=settings, ollama_client=ollama_client, model_name=model_to_pull):
        return True

    try:
        if ollama_client is not None:
            client = ollama_client
        else:
            import ollama
            client = ollama

        logger.info("D4: Pulling model '%s'...", model_to_pull)
        client.pull(model_to_pull)
        logger.info("D4: Model '%s' pulled successfully.", model_to_pull)
        return True

    except Exception as exc:
        logger.error("D4: Failed to pull model: %s", exc)
        return False
