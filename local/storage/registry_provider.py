# local/storage/registry_provider.py
#
# Centralized registry access — the single point of truth for which
# LectureRegistry instance is "live" at any given moment.
#
# DESIGN
# ------
# The module still exposes a default `registry` singleton for backward
# compatibility with code that does `from local.storage.lecture_registry import registry`.
# However, callers that can accept injection (routes, app.py) should use
# `get_registry()` instead, which returns whatever instance has been set via
# `set_registry()`.  Tests call `set_registry(my_isolated_instance)` in their
# fixture setup and `reset_registry()` in teardown — no monkey-patching required.
#
# USAGE
# -----
# Production startup (app.py lifespan):
#   from local.storage.registry_provider import set_registry
#   set_registry(LectureRegistry(registry_file=settings.REGISTRY_FILE))
#
# Tests (fixture):
#   from local.storage.registry_provider import set_registry, reset_registry
#   set_registry(LectureRegistry(registry_file=tmp_path / "test_registry.json"))
#   yield ...
#   reset_registry()
#
# Route/service code:
#   from local.storage.registry_provider import get_registry
#   registry = get_registry()

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_registry_instance: Optional[object] = None  # set to LectureRegistry at runtime
_default_registry: Optional[object] = None   # lazy-initialised default


def get_registry():
    """
    Return the currently active LectureRegistry instance.

    Returns the injected instance if one has been set via set_registry().
    Otherwise falls back to the default module-level instance from
    lecture_registry.py (backward-compatible path).
    """
    global _registry_instance, _default_registry
    if _registry_instance is not None:
        return _registry_instance
    # Lazy default — only import once.
    if _default_registry is None:
        from local.storage.lecture_registry import registry as _default
        _default_registry = _default
        logger.debug("[registry_provider] Using default production registry.")
    return _default_registry


def set_registry(instance) -> None:
    """
    Replace the active registry with the supplied instance.

    Called during application startup with the production instance or during
    test setup with an isolated instance pointing at a tmp directory.
    """
    global _registry_instance
    _registry_instance = instance
    logger.info(
        "[registry_provider] Registry set: %s",
        getattr(instance, "registry_file", "<unknown>"),
    )


def reset_registry() -> None:
    """
    Restore the default registry (clears any injected instance).

    Call this in test teardown to ensure no isolated instance leaks across tests.
    """
    global _registry_instance
    _registry_instance = None
    logger.debug("[registry_provider] Registry reset to default.")
