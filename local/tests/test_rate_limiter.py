# local/tests/test_rate_limiter.py
# Unit tests for the shared LLM rate limiter (no real API calls, no real sleeps).

import threading

import pytest

from local.llm.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    estimate_prompt_tokens,
    get_limiter,
    get_rate_limit_stats,
    reset_rate_limiters,
    reset_rate_limit_stats,
)


class FakeClock:
    """Deterministic clock for limiter tests — never sleeps for real."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


@pytest.fixture(autouse=True)
def _cleanup():
    reset_rate_limiters()
    reset_rate_limit_stats()
    yield
    reset_rate_limiters()
    reset_rate_limit_stats()


def test_burst_within_token_budget_does_not_throttle():
    clock = FakeClock()
    limiter = RateLimiter(
        RateLimitConfig(tokens_per_minute=6000, requests_per_minute=60),
        now=clock.now, sleeper=clock.sleep,
    )
    assert limiter.acquire(1000) == 0.0
    assert limiter.acquire(1000) == 0.0
    assert clock.t == 0.0


def test_token_budget_throttles_above_limit():
    """Token-per-minute must be enforced even when RPM is fine."""
    clock = FakeClock()
    limiter = RateLimiter(
        RateLimitConfig(tokens_per_minute=6000, requests_per_minute=60),
        now=clock.now, sleeper=clock.sleep,
    )
    assert limiter.acquire(4000) == 0.0
    waited = limiter.acquire(4000)  # only 2000 of 4000 tokens left
    assert waited > 0.0
    # refill rate = 6000/60 = 100 tokens/sec; needed ~2000 tokens -> ~20s
    assert waited >= 19.0
    assert clock.t >= 19.0


def test_request_budget_throttles():
    clock = FakeClock()
    limiter = RateLimiter(
        RateLimitConfig(tokens_per_minute=999999, requests_per_minute=2),
        now=clock.now, sleeper=clock.sleep,
    )
    assert limiter.acquire(1) == 0.0
    assert limiter.acquire(1) == 0.0
    waited = limiter.acquire(1)  # third request must wait for a slot
    assert waited > 0.0
    # refill rate = 2/60 per sec -> ~30s for one request slot
    assert waited >= 29.0


def test_get_limiter_shared_per_provider_model():
    a = get_limiter("groq", "gpt-oss-120b")
    b = get_limiter("groq", "gpt-oss-120b")
    c = get_limiter("groq", "other-model")
    assert a is b
    assert a is not c


def test_concurrent_acquire_respects_budget_and_is_thread_safe():
    clock = FakeClock()
    limiter = RateLimiter(
        RateLimitConfig(tokens_per_minute=6000, requests_per_minute=120),
        now=clock.now, sleeper=clock.sleep,
    )
    errors = []

    def worker():
        try:
            for _ in range(3):
                limiter.acquire(1000)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # 24 acquires x 1000 tokens = 24000 tokens vs 6000/min budget.
    # Refill 100 tokens/sec -> total wait >= (24000 - 6000) / 100 = 180s.
    assert clock.t >= 175.0


def test_budget_available_non_blocking():
    clock = FakeClock()
    limiter = RateLimiter(
        RateLimitConfig(tokens_per_minute=6000, requests_per_minute=60),
        now=clock.now, sleeper=clock.sleep,
    )
    assert limiter.budget_available(1000) == 0.0
    limiter.acquire(6000)
    assert limiter.budget_available(1000) > 0.0


def test_zero_budget_rejected():
    with pytest.raises(ValueError):
        RateLimiter(RateLimitConfig(tokens_per_minute=0, requests_per_minute=30))


def test_stats_tracking():
    stats = get_rate_limit_stats()
    stats.record_request()
    stats.record_throttle(2.5)
    stats.record_retry(1.25)
    stats.record_rate_limit()
    stats.record_error("BadRequestError")
    snap = stats.snapshot()
    assert snap["requests"] == 1
    assert snap["throttled_calls"] == 1
    assert snap["throttled_seconds"] == 2.5
    assert snap["retries"] == 1
    assert snap["backoff_seconds"] == 1.25
    assert snap["rate_limit_429s"] == 1
    assert snap["errors"] == {"BadRequestError": 1}


def test_estimate_prompt_tokens():
    assert estimate_prompt_tokens([{"role": "user", "content": "x" * 100}]) == 25
    assert estimate_prompt_tokens([]) == 1
    assert estimate_prompt_tokens([{"role": "user", "content": ""}]) == 1
