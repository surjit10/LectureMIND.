# local/tests/test_d4_ollama.py
# Tests for D4 — Ollama Loader + LLM Service.
# Ollama is fully mocked — no Docker needed.

import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace

from config import LocalSettings
from local.loaders.ollama_loader import check_model_ready
from local.services.llm_service import LLMService


@pytest.fixture
def local_settings():
    return LocalSettings(OLLAMA_MODEL="llama3.1:8b-instruct-q4_K_M")


class TestOllamaLoader:

    def test_model_ready_when_available(self, local_settings):
        """Returns True when the model is listed."""
        mock_client = MagicMock()
        mock_client.list.return_value = {
            "models": [
                {"name": "llama3.1:8b-instruct-q4_K_M"},
                {"name": "mistral:7b"},
            ]
        }

        assert check_model_ready(local_settings=local_settings, ollama_client=mock_client) is True

    def test_model_not_ready_when_missing(self, local_settings):
        """Returns False when the model is not listed."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "mistral:7b"}]}

        assert check_model_ready(local_settings=local_settings, ollama_client=mock_client) is False

    def test_connection_failure_returns_false(self, local_settings):
        """Returns False when Ollama is unreachable."""
        mock_client = MagicMock()
        mock_client.list.side_effect = ConnectionError("Ollama not running")

        assert check_model_ready(local_settings=local_settings, ollama_client=mock_client) is False

    def test_model_ready_with_sdk_response(self, local_settings):
        """Handles ollama SDK-style response with .models attribute."""
        mock_client = MagicMock()
        model_obj = SimpleNamespace(name="llama3.1:8b-instruct-q4_K_M")
        response = SimpleNamespace(models=[model_obj])
        mock_client.list.return_value = response

        assert check_model_ready(local_settings=local_settings, ollama_client=mock_client) is True


class TestLLMService:

    def test_is_ready_delegates_to_check(self, local_settings):
        """LLMService.is_ready() delegates to check_model_ready."""
        mock_client = MagicMock()
        mock_client.list.return_value = {
            "models": [{"name": "llama3.1:8b-instruct-q4_K_M"}]
        }

        service = LLMService(local_settings=local_settings, ollama_client=mock_client)
        assert service.is_ready() is True
        assert service.model_name == "llama3.1:8b-instruct-q4_K_M"

    def test_not_ready_when_model_missing(self, local_settings):
        """LLMService reports not ready when model is missing."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": []}

        service = LLMService(local_settings=local_settings, ollama_client=mock_client)
        assert service.is_ready() is False
