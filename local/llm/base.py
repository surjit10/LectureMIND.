# local/llm/base.py
# Abstract LLM backend interface.
#
# All LLM providers (Ollama, Gemini, OpenAI, Groq, etc.) implement this.
# The answer generator and learning service depend ONLY on this abstraction.
# Provider-specific code is isolated in each backend class.

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class LLMGenerationResult:
    answer: str
    metadata: dict


class LLMBackend(ABC):
    """
    Abstract interface for LLM providers.

    Each concrete backend implements:
      - generate():      produce a text completion from a message list.
      - health_check():  verify the provider is reachable and the model is available.
      - provider_name:   human-readable provider label (e.g. "Ollama", "Google Gemini").
      - model_name:      the model identifier being used.
    """

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Send a chat message list and return the assistant reply as a string.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.

        Returns:
            Assistant reply text, or empty string on failure.
        """
        ...

    def generate_with_metadata(self, messages: List[Dict[str, str]]) -> LLMGenerationResult:
        """
        Send a chat message list and return the answer with token usage metadata.
        Default implementation falls back to generate() with empty metadata.
        """
        answer = self.generate(messages)
        return LLMGenerationResult(answer=answer, metadata={"model": self.model_name, "provider": self.provider_name})

    @abstractmethod
    def health_check(self) -> bool:
        """
        Return True if the provider is reachable and the model is available.
        Must not raise — catch all exceptions internally and return False.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name. Example: 'Ollama', 'Google Gemini'."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier. Example: 'qwen2.5:3b', 'gemini-2.5-flash'."""
        ...
