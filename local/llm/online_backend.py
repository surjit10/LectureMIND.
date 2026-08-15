# local/llm/online_backend.py
# OnlineBackend — remote API-based LLM provider.
#
# Supports any OpenAI-compatible API (OpenAI, Groq, OpenRouter, custom)
# and Google Gemini via its own SDK.
#
# Provider routing is determined by the `provider` field in the config entry.
#
# Rate control: every call flows through the shared limiter in
# local/llm/rate_limiter.py (one budget per provider+model across ALL callers)
# and through a bounded retry loop. HTTP 429 responses honour the provider's
# Retry-After header when present and otherwise use exponential backoff with
# jitter. 429 / transient errors are retried up to max_retries; permanent
# errors (auth, bad request, ...) fail immediately and never loop.

import logging
import random
import time
from typing import Dict, List, Optional, Tuple, Any

from local.llm.base import LLMBackend, LLMGenerationResult
from local.llm.rate_limiter import (
    RateLimitConfig,
    estimate_prompt_tokens,
    get_limiter,
    get_rate_limit_stats,
)

logger = logging.getLogger(__name__)

# Provider identifier constants.  These match what the frontend sends.
PROVIDER_GOOGLE_GEMINI = "google_gemini"
PROVIDER_OPENAI = "openai"
PROVIDER_GROQ = "groq"
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_CUSTOM = "custom"

# Default base URLs for each provider (OpenAI-compatible).
_PROVIDER_BASE_URLS: Dict[str, Optional[str]] = {
    PROVIDER_OPENAI: "https://api.openai.com/v1",
    PROVIDER_GROQ: "https://api.groq.com/openai/v1",
    PROVIDER_OPENROUTER: "https://openrouter.ai/api/v1",
    PROVIDER_ANTHROPIC: None,  # uses Anthropic SDK, not OpenAI-compat
    PROVIDER_GOOGLE_GEMINI: None,  # uses Google SDK
    PROVIDER_CUSTOM: None,  # base_url must be user-supplied
}


class _RateLimited(Exception):
    """A retryable HTTP 429 / rate-limit condition."""

    def __init__(self, message: str = "", retry_after: Optional[float] = None):
        super().__init__(message or "Rate limit exceeded")
        self.retry_after = retry_after


class _TransientError(Exception):
    """A retryable temporary failure (network, timeout, 5xx, 408)."""


class _PermanentError(Exception):
    """A non-retryable failure (auth, bad request, 404, unexpected)."""


def _parse_retry_after(headers: Any) -> Optional[float]:
    """Parse a Retry-After header (seconds or HTTP-date) into seconds."""
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    raw = str(raw).strip()
    # Plain number of seconds.
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    # HTTP-date form.
    try:
        from datetime import datetime, timezone
        from email.utils import parsedate_to_datetime
        parsed = parsedate_to_datetime(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


def _backoff_delay(attempt: int, base_seconds: float, cap_seconds: float) -> float:
    """Bounded exponential backoff with jitter for a retry attempt (0-based)."""
    delay = min(base_seconds * (2 ** attempt), cap_seconds)
    return round(delay * random.uniform(0.5, 1.0), 3)


def _classify(exc: BaseException) -> BaseException:
    """Map an exception onto our internal error semantics.

    - HTTP 429 / rate limit        -> _RateLimited (retryable, Retry-After)
    - timeout / connection / 5xx   -> _TransientError (retryable, backoff)
    - 4xx auth / bad request / 404 -> _PermanentError (no retry)
    - anything unexpected          -> _PermanentError (fail once, never loop)
    """
    try:
        import openai
    except ImportError:
        openai = None

    if openai is not None:
        if isinstance(exc, openai.RateLimitError):
            headers = getattr(exc, "response", None)
            headers = headers.headers if headers is not None else None
            return _RateLimited(str(exc), retry_after=_parse_retry_after(headers))
        if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
            return _TransientError(str(exc))
        if isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", 0)
            if status == 408 or status >= 500:
                return _TransientError(str(exc))
            return _PermanentError(str(exc))

    # String-based fallback for non-OpenAI SDKs (Gemini, Anthropic, custom).
    message = str(exc).lower()
    if "429" in message or "rate limit" in message or "resource_exhausted" in message:
        return _RateLimited(str(exc))
    if "timeout" in message or "connection" in message or "temporarily" in message:
        return _TransientError(str(exc))
    return _PermanentError(str(exc))


class OnlineBackend(LLMBackend):
    """
    Remote (cloud) LLM provider backend.

    Supports:
      - Google Gemini (via google-generativeai SDK)
      - OpenAI (via openai SDK)
      - Groq (OpenAI-compatible via openai SDK with custom base URL)
      - OpenRouter (OpenAI-compatible)
      - Anthropic (via anthropic SDK)
      - Custom OpenAI-compatible endpoints
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        display_name: Optional[str] = None,
        tokens_per_minute: Optional[float] = None,
        requests_per_minute: Optional[float] = None,
    ):
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url or _PROVIDER_BASE_URLS.get(provider)
        self._display_name = display_name or provider
        self._tokens_per_minute = tokens_per_minute
        self._requests_per_minute = requests_per_minute

    @property
    def provider_name(self) -> str:
        return self._display_name

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def tokens_per_minute(self) -> Optional[float]:
        """Optional per-provider token budget override (None = use default)."""
        return self._tokens_per_minute

    @property
    def requests_per_minute(self) -> Optional[float]:
        """Optional per-provider request budget override (None = use default)."""
        return self._requests_per_minute

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Backward compatible generate method."""
        return self.generate_with_metadata(messages).answer

    def generate_with_metadata(self, messages: List[Dict[str, str]]) -> LLMGenerationResult:
        """Generate with shared rate limiting and bounded retries.

        Rate limiting: the shared per-(provider, model) limiter pre-throttles
        so aggregate token/request throughput stays within budget. Retries:
        HTTP 429 -> Retry-After if present else exponential backoff + jitter;
        transient errors (timeout/connection/5xx) -> backoff; permanent errors
        (auth/4xx) -> fail immediately. Never raises; on final failure returns
        an empty answer with error details in metadata, matching the previous
        contract.
        """
        config = RateLimitConfig().with_overrides(
            requests_per_minute=self._requests_per_minute,
            tokens_per_minute=self._tokens_per_minute,
        )
        limiter = get_limiter(self._provider, self._model, config)
        stats = get_rate_limit_stats()
        estimated_tokens = estimate_prompt_tokens(messages) + config.estimated_completion_tokens

        attempt = 0
        total_throttled = 0.0
        while True:
            throttled = limiter.acquire(estimated_tokens)
            if throttled > 0:
                stats.record_throttle(throttled)
                total_throttled += throttled
                logger.info(
                    "OnlineBackend (%s/%s): throttled %.1fs to respect rate limits.",
                    self._provider, self._model, throttled,
                )
            stats.record_request()

            try:
                if self._provider == PROVIDER_GOOGLE_GEMINI:
                    result = self._generate_gemini(messages)
                elif self._provider == PROVIDER_ANTHROPIC:
                    result = self._generate_anthropic(messages)
                else:
                    result = self._generate_openai_compat(messages)
            except Exception as exc:  # noqa: BLE001 — classification below
                classified = _classify(exc)
                if isinstance(classified, _RateLimited):
                    stats.record_rate_limit()
                    if attempt >= config.max_retries:
                        logger.error(
                            "OnlineBackend (%s/%s): rate limited (429) with retries "
                            "exhausted after %d attempts.",
                            self._provider, self._model, attempt + 1,
                        )
                        return LLMGenerationResult(answer="", metadata={
                            "error": "Rate limit exceeded after retries.",
                            "provider": self.provider_name,
                            "model": self._model,
                            "retry_count": attempt,
                            "throttled_seconds": round(total_throttled, 3),
                            "rate_limited": True,
                        })
                    wait = (
                        classified.retry_after
                        if classified.retry_after is not None
                        else _backoff_delay(
                            attempt, config.base_backoff_seconds, config.max_backoff_seconds
                        )
                    )
                    stats.record_retry(wait)
                    logger.warning(
                        "OnlineBackend (%s/%s): rate limited (HTTP 429); retrying in "
                        "%.1fs (attempt %d/%d).",
                        self._provider, self._model, wait, attempt + 1, config.max_retries,
                    )
                    time.sleep(wait)
                    attempt += 1
                    continue

                if isinstance(classified, _TransientError):
                    if attempt >= config.max_retries:
                        logger.error(
                            "OnlineBackend (%s/%s): transient failure (%s) with retries "
                            "exhausted after %d attempts.",
                            self._provider, self._model, type(exc).__name__, attempt + 1,
                        )
                        return LLMGenerationResult(answer="", metadata={
                            "error": str(exc),
                            "provider": self.provider_name,
                            "model": self._model,
                            "retry_count": attempt,
                            "throttled_seconds": round(total_throttled, 3),
                            "rate_limited": False,
                            "transient": True,
                        })
                    wait = _backoff_delay(
                        attempt, config.base_backoff_seconds, config.max_backoff_seconds
                    )
                    stats.record_retry(wait)
                    logger.warning(
                        "OnlineBackend (%s/%s): transient error (%s); retrying in "
                        "%.1fs (attempt %d/%d).",
                        self._provider, self._model, type(exc).__name__,
                        wait, attempt + 1, config.max_retries,
                    )
                    time.sleep(wait)
                    attempt += 1
                    continue

                # Permanent error — fail immediately, never retry.
                stats.record_error(type(exc).__name__)
                logger.error(
                    "OnlineBackend (%s): permanent error (%s): %s",
                    self._provider, type(exc).__name__, exc,
                )
                return LLMGenerationResult(answer="", metadata={
                    "error": str(exc),
                    "provider": self.provider_name,
                    "model": self._model,
                    "retry_count": attempt,
                    "throttled_seconds": round(total_throttled, 3),
                    "rate_limited": False,
                })

            # Success — preserve existing metadata and add rate-limit diagnostics.
            metadata = dict(result.metadata or {})
            metadata.setdefault("provider", self.provider_name)
            metadata.setdefault("model", self._model)
            metadata["retry_count"] = attempt
            metadata["throttled_seconds"] = round(total_throttled, 3)
            metadata["rate_limited"] = False
            return LLMGenerationResult(answer=result.answer, metadata=metadata)

    def health_check(self) -> bool:
        """Attempt a minimal API call to verify the provider is reachable."""
        try:
            success, msg, _ = self.test_connection(self._provider, self._api_key, self._base_url)
            return success
        except Exception as exc:
            logger.error("OnlineBackend (%s): health check failed: %s", self._provider, exc)
            return False

    @classmethod
    def test_connection(
        cls, provider: str, api_key: str, base_url: Optional[str] = None
    ) -> Tuple[bool, str, List[str]]:
        """
        Validate the API key and fetch available models.
        Returns (success, message, list_of_models).
        """
        base_url = base_url or _PROVIDER_BASE_URLS.get(provider)
        try:
            if provider == PROVIDER_GOOGLE_GEMINI:
                models = cls._list_models_gemini(api_key)
            elif provider == PROVIDER_ANTHROPIC:
                models = cls._list_models_anthropic(api_key)
            else:
                # OpenAI compatible
                models = cls._list_models_openai(provider, api_key, base_url)

            return True, "Connection successful.", models
        except Exception as exc:
            # Try to return a friendly error message from the exception
            err_msg = str(exc)
            if "401" in err_msg or "Unauthorized" in err_msg or "authentication" in err_msg.lower() or "api key" in err_msg.lower():
                err_msg = "Invalid API key."
            elif "timeout" in err_msg.lower():
                err_msg = "Network timeout."
            elif "429" in err_msg or "rate limit" in err_msg.lower():
                err_msg = "Rate limit exceeded."
            elif "Connection error" in err_msg:
                err_msg = "Provider unreachable."
            else:
                err_msg = f"Connection failed: {err_msg}"

            return False, err_msg, []

    # ------------------------------------------------------------------
    # Model Discovery methods
    # ------------------------------------------------------------------

    @classmethod
    def _list_models_openai(cls, provider: str, api_key: str, base_url: Optional[str]) -> List[str]:
        from openai import OpenAI
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        # Use a short timeout for health checks
        response = client.models.list(timeout=10.0)

        models = [m.id for m in response.data]
        # Filter for typical LLM models if provider is pure OpenAI to avoid huge list of non-LLMs
        if provider == PROVIDER_OPENAI:
            models = [m for m in models if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3")]
            models.sort(reverse=True)
        return models

    @classmethod
    def _list_models_gemini(cls, api_key: str) -> List[str]:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                models.append(m.name.replace("models/", ""))
        return models

    @classmethod
    def _list_models_anthropic(cls, api_key: str) -> List[str]:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.models.list(timeout=10.0)
        return [m.id for m in response.data]

    # ------------------------------------------------------------------
    # Private generator methods — one per SDK family
    # ------------------------------------------------------------------

    def _generate_openai_compat(self, messages: List[Dict[str, str]]) -> LLMGenerationResult:
        """OpenAI SDK call — works for OpenAI, Groq, OpenRouter, Custom.

        max_retries=0 disables the SDK's own retry loop so the bounded retry
        policy in generate_with_metadata() is the single retry layer — the
        SDK default (max_retries=2) would otherwise multiply requests and
        make rate limiting worse.
        """
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("OnlineBackend: 'openai' package not installed. Run: pip install openai")
            return LLMGenerationResult(answer="", metadata={})

        kwargs: Dict = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        kwargs["max_retries"] = 0

        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
        )

        metadata = {"model": self._model, "provider": self.provider_name}
        if hasattr(response, "usage") and response.usage:
            metadata["prompt_tokens"] = response.usage.prompt_tokens
            metadata["completion_tokens"] = response.usage.completion_tokens

        return LLMGenerationResult(answer=response.choices[0].message.content or "", metadata=metadata)

    def _generate_gemini(self, messages: List[Dict[str, str]]) -> LLMGenerationResult:
        """Google Gemini SDK call."""
        try:
            import google.generativeai as genai
        except ImportError:
            logger.error(
                "OnlineBackend: 'google-generativeai' package not installed. "
                "Run: pip install google-generativeai"
            )
            return LLMGenerationResult(answer="", metadata={})

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._model)

        # Convert OpenAI-style messages to Gemini format.
        # Gemini uses 'user' / 'model' roles (not 'assistant').
        gemini_history = []
        system_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                gemini_history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_history.append({"role": "model", "parts": [content]})

        # Prepend system prompt to first user message if present.
        if system_parts and gemini_history and gemini_history[0]["role"] == "user":
            system_text = "\n".join(system_parts)
            gemini_history[0]["parts"] = [system_text + "\n\n" + gemini_history[0]["parts"][0]]

        if not gemini_history:
            return LLMGenerationResult(answer="", metadata={})

        # For a single-turn exchange use generate_content; multi-turn uses chat.
        if len(gemini_history) == 1:
            response = model.generate_content(gemini_history[0]["parts"][0])
            metadata = {"model": self._model, "provider": self.provider_name}
            if hasattr(response, "usage_metadata"):
                metadata["prompt_tokens"] = response.usage_metadata.prompt_token_count
                metadata["completion_tokens"] = response.usage_metadata.candidates_token_count
            return LLMGenerationResult(answer=response.text or "", metadata=metadata)
        else:
            chat = model.start_chat(history=gemini_history[:-1])
            last_user_msg = gemini_history[-1]["parts"][0]
            response = chat.send_message(last_user_msg)
            metadata = {"model": self._model, "provider": self.provider_name}
            if hasattr(response, "usage_metadata"):
                metadata["prompt_tokens"] = response.usage_metadata.prompt_token_count
                metadata["completion_tokens"] = response.usage_metadata.candidates_token_count
            return LLMGenerationResult(answer=response.text or "", metadata=metadata)

    def _generate_anthropic(self, messages: List[Dict[str, str]]) -> LLMGenerationResult:
        """Anthropic Claude SDK call."""
        try:
            import anthropic
        except ImportError:
            logger.error(
                "OnlineBackend: 'anthropic' package not installed. "
                "Run: pip install anthropic"
            )
            return LLMGenerationResult(answer="", metadata={})

        client = anthropic.Anthropic(api_key=self._api_key)

        # Separate system messages from the chat history.
        system_text = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs: Dict = {
            "model": self._model,
            "max_tokens": 2048,
            "messages": chat_messages,
        }
        if system_text:
            kwargs["system"] = system_text

        response = client.messages.create(**kwargs)

        metadata = {"model": self._model, "provider": self.provider_name}
        if hasattr(response, "usage"):
            metadata["prompt_tokens"] = getattr(response.usage, "input_tokens", 0)
            metadata["completion_tokens"] = getattr(response.usage, "output_tokens", 0)

        return LLMGenerationResult(answer=(response.content[0].text if response.content else ""), metadata=metadata)
