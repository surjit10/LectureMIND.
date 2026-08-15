# local/llm/rate_limiter.py
# Shared rate-control layer for online LLM providers.
#
# Every online LLM call (OnlineBackend, used by the answer generator, the
# learning service, the benchmark, and scripts) flows through a single shared
# limiter so that aggregate token / request throughput never exceeds the
# provider budget — even when several callers or threads issue requests at
# once.
#
# Two budgets are enforced per (provider, model):
#   * requests_per_minute — caps call frequency (the provider's RPM limit).
#   * tokens_per_minute   — caps total tokens per minute. This is the budget
#     that actually protects token-per-minute limits: several long-context
#     calls inside one minute can burn thousands of tokens without exceeding
#     the RPM limit (e.g. 30 RPM x ~4K tokens/call = 120K TPM >> a 6-7K TPM
#     provider limit). The limiter therefore accounts for an estimate of the
#     prompt size plus an estimated completion size for every call.
#
# The limiter PRE-THROTTLES: it waits before issuing the request until both
# buckets have room, so bursts are smoothed out client-side and the provider's
# limit is respected without hammering it with 429 responses.
#
# Budgets are configurable per provider via data/llm_config.json
# (tokens_per_minute / requests_per_minute on the provider entry), or
# globally via environment variables (see RateLimitConfig). Defaults are
# conservative and documented — set them to match the provider plan actually
# in use.
#
# Thread-safety: acquire() is guarded by a per-(provider, model) lock, so
# concurrent callers (e.g. FastAPI threadpool) share one budget instead of
# each creating an independent one.

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    """Read a positive float from the environment, falling back to default."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(raw)
        return value if value > 0 else default
    except ValueError:
        logger.warning("Invalid value for %s=%r — using default %.1f", name, raw, default)
        return default


@dataclass
class RateLimitConfig:
    """Budget + retry policy for one online LLM provider.

    Defaults are a conservative OpenAI-compatible provider plan. Override per
    provider with ``tokens_per_minute`` / ``requests_per_minute`` on the
    provider entry in ``data/llm_config.json``, or globally via env:
      LLM_REQUESTS_PER_MINUTE        (default 30)
      LLM_TOKENS_PER_MINUTE          (default 6000)
      LLM_MAX_RETRIES                (default 4)
      LLM_BASE_BACKOFF_SECONDS       (default 1.0)
      LLM_MAX_BACKOFF_SECONDS        (default 30.0)
      LLM_ESTIMATED_COMPLETION_TOKENS (default 1024)
    """

    requests_per_minute: float = field(
        default_factory=lambda: _env_float("LLM_REQUESTS_PER_MINUTE", 30.0)
    )
    tokens_per_minute: float = field(
        default_factory=lambda: _env_float("LLM_TOKENS_PER_MINUTE", 6000.0)
    )
    max_retries: int = field(
        default_factory=lambda: int(_env_float("LLM_MAX_RETRIES", 4))
    )
    base_backoff_seconds: float = field(
        default_factory=lambda: _env_float("LLM_BASE_BACKOFF_SECONDS", 1.0)
    )
    max_backoff_seconds: float = field(
        default_factory=lambda: _env_float("LLM_MAX_BACKOFF_SECONDS", 30.0)
    )
    estimated_completion_tokens: int = field(
        default_factory=lambda: int(_env_float("LLM_ESTIMATED_COMPLETION_TOKENS", 1024))
    )

    def with_overrides(
        self,
        requests_per_minute: Optional[float] = None,
        tokens_per_minute: Optional[float] = None,
    ) -> "RateLimitConfig":
        """Return a copy with per-provider budget overrides applied."""
        cfg = RateLimitConfig()
        cfg.requests_per_minute = float(
            requests_per_minute if requests_per_minute is not None else self.requests_per_minute
        )
        cfg.tokens_per_minute = float(
            tokens_per_minute if tokens_per_minute is not None else self.tokens_per_minute
        )
        cfg.max_retries = self.max_retries
        cfg.base_backoff_seconds = self.base_backoff_seconds
        cfg.max_backoff_seconds = self.max_backoff_seconds
        cfg.estimated_completion_tokens = self.estimated_completion_tokens
        return cfg


def estimate_prompt_tokens(messages: List[Dict[str, str]]) -> int:
    """Rough token estimate for a chat message list (~4 chars/token)."""
    total = 0
    for message in messages or []:
        content = message.get("content", "") if isinstance(message, dict) else ""
        total += len(str(content))
    return max(1, total // 4)


class _Bucket:
    """A simple token bucket with continuous refill."""

    __slots__ = ("capacity", "refill_per_sec", "tokens", "last_refill")

    def __init__(self, capacity: float, refill_per_sec: float, now: float):
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self.tokens = capacity
        self.last_refill = now

    def refill(self, now: float) -> None:
        if now > self.last_refill:
            self.tokens = min(
                self.capacity,
                self.tokens + (now - self.last_refill) * self.refill_per_sec,
            )
            self.last_refill = now


# Tolerance for floating-point comparisons in the buckets, and the minimum
# sleep step that guarantees the refill clock always makes forward progress.
_EPS = 1e-9
_MIN_STEP = 1e-3


class RateLimiter:
    """Thread-safe token/request bucket limiter for one (provider, model).

    ``now`` and ``sleeper`` are injectable so tests can use a fake clock and
    never touch real sleeping.
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        now=time.monotonic,
        sleeper=time.sleep,
    ):
        self._config = config or RateLimitConfig()
        if self._config.requests_per_minute <= 0 or self._config.tokens_per_minute <= 0:
            raise ValueError("RateLimitConfig budgets must be positive.")
        self._now = now
        self._sleep = sleeper
        self._lock = threading.Lock()
        start = now()
        self._requests = _Bucket(
            self._config.requests_per_minute,
            self._config.requests_per_minute / 60.0,
            start,
        )
        self._tokens = _Bucket(
            self._config.tokens_per_minute,
            self._config.tokens_per_minute / 60.0,
            start,
        )

    def acquire(self, estimated_tokens: float = 1.0) -> float:
        """Block until both buckets have room, consume, return wait seconds."""
        estimated_tokens = max(1.0, float(estimated_tokens))
        waited = 0.0
        with self._lock:
            while True:
                now = self._now()
                self._requests.refill(now)
                self._tokens.refill(now)
                # Epsilon tolerance: floating-point refill can land a fraction
                # of an ulp below the requirement; without it the loop would
                # spin forever on sub-representable waits (see MIN_STEP below).
                if (
                    self._requests.tokens >= 1.0 - _EPS
                    and self._tokens.tokens >= estimated_tokens - _EPS
                ):
                    self._requests.tokens -= 1.0
                    self._tokens.tokens -= estimated_tokens
                    return waited
                # Per-axis wait is only positive when that axis is actually
                # constrained; we must wait for BOTH buckets to be ready, so
                # the sleep is the MAX of the two (never negative — a bucket
                # with spare capacity adds zero wait).
                req_wait = (
                    0.0
                    if self._requests.tokens >= 1.0 - _EPS
                    else (1.0 - self._requests.tokens) / self._requests.refill_per_sec
                    if self._requests.refill_per_sec > 0
                    else float("inf")
                )
                tok_wait = (
                    0.0
                    if self._tokens.tokens >= estimated_tokens - _EPS
                    else (estimated_tokens - self._tokens.tokens) / self._tokens.refill_per_sec
                    if self._tokens.refill_per_sec > 0
                    else float("inf")
                )
                wait = max(req_wait, tok_wait)
                if wait == float("inf"):
                    raise RuntimeError("Rate limiter misconfigured: zero budgets")
                # Ensure forward progress: a sub-representable wait (e.g. 1e-17s
                # from float dust) advances the clock by less than one ulp of
                # the token balance, the refill underflows, and the loop would
                # never terminate. Clamp to a minimum step, and cap the step so
                # the loop re-checks and stays accurate.
                wait = min(max(wait, _MIN_STEP), 5.0)
                self._sleep(wait)
                waited += wait

    def budget_available(self, estimated_tokens: float = 1.0) -> float:
        """Non-blocking check: seconds until capacity for this request, 0 if ready."""
        now = self._now()
        with self._lock:
            self._requests.refill(now)
            self._tokens.refill(now)
            if (
                self._requests.tokens >= 1.0 - _EPS
                and self._tokens.tokens >= estimated_tokens - _EPS
            ):
                return 0.0
            req_wait = (
                0.0
                if self._requests.tokens >= 1.0 - _EPS
                else (1.0 - self._requests.tokens) / self._requests.refill_per_sec
                if self._requests.refill_per_sec > 0
                else float("inf")
            )
            tok_wait = (
                0.0
                if self._tokens.tokens >= estimated_tokens - _EPS
                else (estimated_tokens - self._tokens.tokens) / self._tokens.refill_per_sec
                if self._tokens.refill_per_sec > 0
                else float("inf")
            )
            return max(req_wait, tok_wait)


# ---------------------------------------------------------------------------
# Module-level shared limiter registry: one limiter per (provider, model).
# ---------------------------------------------------------------------------

_limiters: Dict[Tuple[str, str], RateLimiter] = {}
_limiters_lock = threading.Lock()


def get_limiter(
    provider: str,
    model: str,
    config: Optional[RateLimitConfig] = None,
) -> RateLimiter:
    """Return the shared limiter for (provider, model).

    All OnlineBackend instances (and therefore all callers) share this
    instance, so budgets are enforced globally rather than per caller.
    """
    key = (provider or "unknown", model or "unknown")
    with _limiters_lock:
        limiter = _limiters.get(key)
        if limiter is None:
            limiter = RateLimiter(config=config)
            _limiters[key] = limiter
        return limiter


def reset_rate_limiters() -> None:
    """Clear the shared limiter registry (tests / config reload)."""
    with _limiters_lock:
        _limiters.clear()


# ---------------------------------------------------------------------------
# Diagnostics — retry count and throttling time for dashboards/reports.
# ---------------------------------------------------------------------------

class RateLimitStats:
    """Process-wide counters for rate limiting and retry behaviour."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests = 0
        self._throttled_calls = 0
        self._throttled_seconds = 0.0
        self._retries = 0
        self._backoff_seconds = 0.0
        self._rate_limit_429s = 0
        self._errors: Dict[str, int] = {}

    def record_request(self) -> None:
        with self._lock:
            self._requests += 1

    def record_throttle(self, seconds: float) -> None:
        with self._lock:
            self._throttled_calls += 1
            self._throttled_seconds += seconds

    def record_retry(self, backoff_seconds: float) -> None:
        with self._lock:
            self._retries += 1
            self._backoff_seconds += backoff_seconds

    def record_rate_limit(self) -> None:
        with self._lock:
            self._rate_limit_429s += 1

    def record_error(self, kind: str) -> None:
        with self._lock:
            self._errors[kind] = self._errors.get(kind, 0) + 1

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "requests": self._requests,
                "throttled_calls": self._throttled_calls,
                "throttled_seconds": round(self._throttled_seconds, 3),
                "retries": self._retries,
                "backoff_seconds": round(self._backoff_seconds, 3),
                "rate_limit_429s": self._rate_limit_429s,
                "errors": dict(self._errors),
            }


_stats = RateLimitStats()


def get_rate_limit_stats() -> RateLimitStats:
    """Return the process-wide rate-limit statistics singleton."""
    return _stats


def reset_rate_limit_stats() -> None:
    """Reset the statistics singleton (tests)."""
    global _stats
    _stats = RateLimitStats()
