# local/tests/test_online_backend_retry.py
# Unit tests for OnlineBackend rate-limit handling and retry policy.
# All API calls are mocked — no real provider traffic, no real sleeping.

import threading
from unittest.mock import MagicMock, patch

import httpx
import pytest

from local.llm.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    reset_rate_limiters,
    reset_rate_limit_stats,
)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def _resp(status, headers=None):
    return httpx.Response(
        status_code=status,
        headers=headers or {},
        request=httpx.Request("POST", "http://provider"),
    )


def _rate_limit_error(retry_after=None):
    import openai
    headers = {}
    if retry_after is not None:
        headers["retry-after"] = str(retry_after)
    return openai.RateLimitError("Rate limit", response=_resp(429, headers), body=None)


def _auth_error():
    import openai
    return openai.AuthenticationError("bad key", response=_resp(401), body=None)


def _connection_error():
    import openai
    return openai.APIConnectionError(request=httpx.Request("POST", "http://provider"))


def _timeout_error():
    import openai
    return openai.APITimeoutError(request=httpx.Request("POST", "http://provider"))


def _ok_response():
    message = MagicMock()
    message.content = "42"
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return response


def _make_backend(**kwargs):
    from local.llm.online_backend import OnlineBackend
    params = dict(
        provider="groq",
        model="openai/gpt-oss-120b",
        api_key="sk-test",
        base_url="https://api.groq.com/openai/v1",
        # Huge budgets so acquire() never throttles in retry tests.
        tokens_per_minute=1e9,
        requests_per_minute=1e9,
    )
    params.update(kwargs)
    return OnlineBackend(**params)


@pytest.fixture(autouse=True)
def _cleanup():
    reset_rate_limiters()
    reset_rate_limit_stats()
    yield
    reset_rate_limiters()
    reset_rate_limit_stats()


@pytest.fixture
def limiter_patch():
    """Deterministic shared limiter with a huge budget: acquire() never
    throttles and never sleeps for real, so retry tests only exercise the
    retry policy."""
    clock = FakeClock()
    fake = RateLimiter(
        RateLimitConfig(tokens_per_minute=1e9, requests_per_minute=1e9),
        now=clock.now, sleeper=clock.sleep,
    )
    with patch("local.llm.online_backend.get_limiter", return_value=fake):
        yield fake


# ---------------------------------------------------------------------------
# 429 handling
# ---------------------------------------------------------------------------

def test_429_retried_then_success(limiter_patch):
    backend = _make_backend()
    calls = {"n": 0}

    def fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _rate_limit_error()
        return _ok_response()

    with patch("openai.OpenAI") as mock_openai, patch("time.sleep") as mock_sleep, \
         patch("local.llm.online_backend.random.uniform", return_value=1.0):
        mock_openai.return_value.chat.completions.create.side_effect = fake_create
        result = backend.generate_with_metadata([{"role": "user", "content": "hi"}])

    assert result.answer == "42"
    assert calls["n"] == 2
    assert result.metadata["retry_count"] == 1
    assert result.metadata["rate_limited"] is False
    assert result.metadata["throttled_seconds"] == 0.0
    # One backoff sleep on the first attempt (deterministic jitter = x1.0).
    mock_sleep.assert_called_once_with(1.0)


def test_retry_after_header_respected(limiter_patch):
    backend = _make_backend()
    calls = {"n": 0}

    def fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _rate_limit_error(retry_after=3)
        return _ok_response()

    with patch("openai.OpenAI") as mock_openai, patch("time.sleep") as mock_sleep:
        mock_openai.return_value.chat.completions.create.side_effect = fake_create
        result = backend.generate_with_metadata([{"role": "user", "content": "hi"}])

    assert result.answer == "42"
    mock_sleep.assert_called_once_with(3.0)  # provider Retry-After wins


def test_max_retry_exhaustion_no_unbounded_loop(limiter_patch):
    backend = _make_backend()
    with patch("openai.OpenAI") as mock_openai, patch("time.sleep") as mock_sleep:
        mock_openai.return_value.chat.completions.create.side_effect = _rate_limit_error()
        result = backend.generate_with_metadata([{"role": "user", "content": "hi"}])

    assert result.answer == ""
    assert result.metadata.get("rate_limited") is True
    assert "rate limit" in result.metadata["error"].lower()
    # initial call + max_retries retries (default 4) — bounded, no unbounded loop
    assert mock_openai.return_value.chat.completions.create.call_count == 5
    assert mock_sleep.call_count == 4


def test_max_retries_configurable(limiter_patch, monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    backend = _make_backend()
    with patch("openai.OpenAI") as mock_openai, patch("time.sleep") as mock_sleep:
        mock_openai.return_value.chat.completions.create.side_effect = _rate_limit_error()
        result = backend.generate_with_metadata([{"role": "user", "content": "hi"}])

    assert result.answer == ""
    assert mock_openai.return_value.chat.completions.create.call_count == 3
    assert mock_sleep.call_count == 2


def test_429_does_not_duplicate_unbounded_extraction(limiter_patch):
    """A caller loop (e.g. extraction) gets one bounded attempt per call."""
    backend = _make_backend()
    with patch("openai.OpenAI") as mock_openai, patch("time.sleep"):
        mock_openai.return_value.chat.completions.create.side_effect = _rate_limit_error()
        out = [
            backend.generate(messages=[{"role": "user", "content": "p"}])
            for _ in range(3)
        ]

    assert out == ["", "", ""]
    # 3 extraction calls x (1 initial + 4 retries) = 15 API calls, never more
    assert mock_openai.return_value.chat.completions.create.call_count == 15


# ---------------------------------------------------------------------------
# Transient vs permanent error semantics
# ---------------------------------------------------------------------------

def test_connection_error_retried_then_success(limiter_patch):
    backend = _make_backend()
    calls = {"n": 0}

    def fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _connection_error()
        return _ok_response()

    with patch("openai.OpenAI") as mock_openai, patch("time.sleep") as mock_sleep:
        mock_openai.return_value.chat.completions.create.side_effect = fake_create
        result = backend.generate_with_metadata([{"role": "user", "content": "hi"}])

    assert result.answer == "42"
    assert calls["n"] == 2
    mock_sleep.assert_called_once()


def test_timeout_retried_then_success(limiter_patch):
    backend = _make_backend()
    calls = {"n": 0}

    def fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _timeout_error()
        return _ok_response()

    with patch("openai.OpenAI") as mock_openai, patch("time.sleep"):
        mock_openai.return_value.chat.completions.create.side_effect = fake_create
        result = backend.generate_with_metadata([{"role": "user", "content": "hi"}])

    assert result.answer == "42"
    assert calls["n"] == 2


def test_permanent_auth_error_not_retried(limiter_patch):
    backend = _make_backend()
    with patch("openai.OpenAI") as mock_openai, patch("time.sleep") as mock_sleep:
        mock_openai.return_value.chat.completions.create.side_effect = _auth_error()
        result = backend.generate_with_metadata([{"role": "user", "content": "hi"}])

    assert result.answer == ""
    assert "bad key" in result.metadata["error"]
    assert result.metadata["rate_limited"] is False
    assert mock_openai.return_value.chat.completions.create.call_count == 1  # no retry
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Successful calls unchanged
# ---------------------------------------------------------------------------

def test_successful_call_unchanged(limiter_patch):
    backend = _make_backend()
    with patch("openai.OpenAI") as mock_openai, patch("time.sleep") as mock_sleep:
        mock_openai.return_value.chat.completions.create.return_value = _ok_response()
        result = backend.generate_with_metadata([{"role": "user", "content": "hi"}])

    assert result.answer == "42"
    assert result.metadata["provider"] == "groq"
    assert result.metadata["model"] == "openai/gpt-oss-120b"
    assert result.metadata["prompt_tokens"] == 10
    assert result.metadata["completion_tokens"] == 5
    assert result.metadata["retry_count"] == 0
    assert result.metadata["rate_limited"] is False
    mock_sleep.assert_not_called()


def test_sdk_automatic_retries_disabled():
    """The OpenAI SDK's own retry loop must be off so retries never multiply."""
    backend = _make_backend()
    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _ok_response()
        backend.generate_with_metadata([{"role": "user", "content": "hi"}])
        assert mock_openai.call_args.kwargs.get("max_retries") == 0


# ---------------------------------------------------------------------------
# Concurrency through the shared limiter
# ---------------------------------------------------------------------------

def test_concurrent_requests_through_shared_limiter():
    """Two backends (as if two callers) share one budget; no request is lost."""
    from local.llm.online_backend import OnlineBackend

    clock = FakeClock()
    shared = RateLimiter(
        RateLimitConfig(tokens_per_minute=3000, requests_per_minute=60),
        now=clock.now, sleeper=clock.sleep,
    )
    b1 = OnlineBackend(
        provider="groq", model="m", api_key="k", tokens_per_minute=3000
    )
    b2 = OnlineBackend(
        provider="groq", model="m", api_key="k", tokens_per_minute=3000
    )

    with patch("local.llm.online_backend.get_limiter", return_value=shared), \
         patch("openai.OpenAI") as mock_openai, patch("time.sleep"):
        mock_openai.return_value.chat.completions.create.return_value = _ok_response()
        results = []

        def worker(backend):
            results.append(
                backend.generate_with_metadata(
                    [{"role": "user", "content": "x" * 2000}]
                ).answer
            )

        threads = [threading.Thread(target=worker, args=(b,)) for b in (b1, b2) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert results == ["42"] * 6
    assert mock_openai.return_value.chat.completions.create.call_count == 6
    # ~1500 tokens/call x 6 = ~9000 tokens vs 3000 budget -> throttled
    assert clock.t > 0.0
