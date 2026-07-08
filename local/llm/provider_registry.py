# local/llm/provider_registry.py
# Persistent store for LLM provider configuration.
#
# Stores:
#   - inference_mode:    "offline" | "online"
#   - local_model:       active Ollama model name
#   - active_provider_id: ID of the active online provider (or None)
#   - providers:         list of configured online provider entries
#
# API keys are stored in the config file (local filesystem only).
# They are NEVER returned through the API — the API replaces them with "***".
#
# Config file: data/llm_config.json

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_FILE = _PROJECT_ROOT / "data" / "llm_config.json"

_DEFAULT_CONFIG: Dict[str, Any] = {
    "inference_mode": "offline",
    "local_model": "qwen2.5:3b",
    "active_provider_id": None,
    "providers": [],
}


class ProviderRegistry:
    """
    Persistent registry of LLM provider configurations.

    Thread-safety: single-process single-thread (FastAPI with uvicorn default
    worker).  For multi-worker deployments, wrap _save/_load with a file lock.
    """

    def __init__(self, config_file: str | Path = _CONFIG_FILE):
        self._file = Path(config_file)
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._cfg: Dict[str, Any] = {}
        self._last_loaded_mtime: Optional[float] = None
        self._load()

    def _sync_if_modified(self) -> None:
        """Reload configuration from disk if it was modified by another process."""
        import os
        if not self._file.exists():
            return
            
        try:
            current_mtime = os.path.getmtime(self._file)
        except OSError:
            return

        old_mtime = self._last_loaded_mtime
        reloaded = False
        
        if old_mtime is None or current_mtime > old_mtime:
            self._load()
            reloaded = True
            
        logger.info(
            "[ProviderRegistry]\n"
            "PID=%s\n"
            "Registry=%s\n"
            "File=%s\n"
            "Reload=%s\n"
            "ActiveProvider=%s\n"
            "mtime(old)=%s\n"
            "mtime(new)=%s\n",
            os.getpid(),
            id(self),
            self._file,
            reloaded,
            self._cfg.get("active_provider_id"),
            old_mtime,
            current_mtime
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._file.exists():
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # Forward-compat: merge defaults for any missing keys.
                self._cfg = {**_DEFAULT_CONFIG, **loaded}
                # Ensure providers list exists.
                if not isinstance(self._cfg.get("providers"), list):
                    self._cfg["providers"] = []
                import os
                self._last_loaded_mtime = os.path.getmtime(self._file)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "[llm_config] Failed to load config — using defaults: %s", exc
                )
                self._cfg = dict(_DEFAULT_CONFIG)
                self._last_loaded_mtime = None
        else:
            logger.info("[llm_config] No config file found — starting with defaults.")
            self._cfg = dict(_DEFAULT_CONFIG)
            self._last_loaded_mtime = None

    def _save(self) -> None:
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(self._cfg, f, indent=2)
            import os
            self._last_loaded_mtime = os.path.getmtime(self._file)
        except OSError as exc:
            logger.error("[llm_config] Failed to save config: %s", exc)

    # ------------------------------------------------------------------
    # Global settings
    # ------------------------------------------------------------------

    def get_inference_mode(self) -> str:
        self._sync_if_modified()
        return self._cfg.get("inference_mode", "offline")

    def set_inference_mode(self, mode: str) -> None:
        if mode not in ("offline", "online"):
            raise ValueError(f"Invalid inference_mode: '{mode}'. Must be 'offline' or 'online'.")
        self._cfg["inference_mode"] = mode
        self._save()

    def get_local_model(self) -> str:
        self._sync_if_modified()
        return self._cfg.get("local_model", "qwen2.5:3b")

    def set_local_model(self, model: str) -> None:
        if not model.strip():
            raise ValueError("local_model must not be empty.")
        self._cfg["local_model"] = model.strip()
        self._save()

    def get_active_provider_id(self) -> Optional[str]:
        self._sync_if_modified()
        return self._cfg.get("active_provider_id")

    def set_active_provider_id(self, provider_id: Optional[str]) -> None:
        if provider_id is not None:
            if not self._find_provider(provider_id):
                raise ValueError(f"Provider '{provider_id}' not found.")
        self._cfg["active_provider_id"] = provider_id
        self._save()

    # ------------------------------------------------------------------
    # Provider CRUD
    # ------------------------------------------------------------------

    def _find_provider(self, provider_id: str) -> Optional[Dict[str, Any]]:
        self._sync_if_modified()
        for p in self._cfg["providers"]:
            if p.get("id") == provider_id:
                return p
        return None

    def list_providers(self) -> List[Dict[str, Any]]:
        """Return all providers with API keys masked."""
        self._sync_if_modified()
        return [self._mask_key(dict(p)) for p in self._cfg["providers"]]

    def get_provider_raw(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Return a provider dict WITH the real API key (internal use only)."""
        # _find_provider will trigger _sync_if_modified()
        return self._find_provider(provider_id)

    def add_provider(
        self,
        provider: str,
        model: str,
        api_key: str,
        display_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a new online provider. Returns masked entry."""
        if not provider.strip():
            raise ValueError("provider must not be empty.")
        if not model.strip():
            raise ValueError("model must not be empty.")
        if not api_key.strip():
            raise ValueError("api_key must not be empty.")

        entry: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "provider": provider.strip(),
            "model": model.strip(),
            "api_key": api_key.strip(),
            "display_name": (display_name or provider).strip(),
            "base_url": base_url.strip() if base_url else None,
        }
        self._cfg["providers"].append(entry)
        
        # If this is the first provider or no active provider is set, set it as active
        if not self._cfg.get("active_provider_id"):
            self._cfg["active_provider_id"] = entry["id"]
            
        self._save()
        logger.info(
            "[llm_config] Provider added: %s (%s / %s)",
            entry["id"], entry["provider"], entry["model"],
        )
        return self._mask_key(dict(entry))

    def update_provider(
        self,
        provider_id: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        display_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a provider entry. Only supplied fields are changed."""
        entry = self._find_provider(provider_id)
        if not entry:
            raise KeyError(f"Provider '{provider_id}' not found.")
        if model is not None:
            if not model.strip():
                raise ValueError("model must not be empty.")
            entry["model"] = model.strip()
        if api_key is not None:
            if not api_key.strip():
                raise ValueError("api_key must not be empty.")
            entry["api_key"] = api_key.strip()
        if display_name is not None:
            entry["display_name"] = display_name.strip()
        if base_url is not None:
            entry["base_url"] = base_url.strip() if base_url.strip() else None
        self._save()
        return self._mask_key(dict(entry))

    def delete_provider(self, provider_id: str) -> None:
        """Remove a provider. Clears active_provider_id if it was this one."""
        before = len(self._cfg["providers"])
        self._cfg["providers"] = [
            p for p in self._cfg["providers"] if p.get("id") != provider_id
        ]
        if len(self._cfg["providers"]) == before:
            raise KeyError(f"Provider '{provider_id}' not found.")
        if self._cfg.get("active_provider_id") == provider_id:
            # Fall back to another provider if available
            if self._cfg["providers"]:
                self._cfg["active_provider_id"] = self._cfg["providers"][0]["id"]
            else:
                # No providers left, safely revert to offline mode
                self._cfg["active_provider_id"] = None
                self._cfg["inference_mode"] = "offline"
        self._save()
        logger.info("[llm_config] Provider deleted: %s", provider_id)

    # ------------------------------------------------------------------
    # LLM backend factory
    # ------------------------------------------------------------------

    def get_active_backend(self) -> "Any":  # -> LLMBackend
        """
        Return the currently active LLMBackend instance.

        - If inference_mode == "offline" → OllamaBackend with local_model.
        - If inference_mode == "online"  → OnlineBackend for active_provider_id.
        - Falls back to OllamaBackend if online mode has no active provider.
        """
        from local.llm.ollama_backend import OllamaBackend
        from local.llm.online_backend import OnlineBackend

        # Synchronize happens inside get_inference_mode and get_active_provider_id
        mode = self.get_inference_mode()
        pid = self.get_active_provider_id()

        if mode == "online":
            if pid:
                entry = self.get_provider_raw(pid)
                if entry:
                    return OnlineBackend(
                        provider=entry["provider"],
                        model=entry["model"],
                        api_key=entry["api_key"],
                        base_url=entry.get("base_url"),
                        display_name=entry.get("display_name"),
                    )
            logger.warning(
                "[llm_config] Online mode selected but no active provider configured. "
                "Falling back to Ollama."
            )

        # Default / fallback: Ollama.
        return OllamaBackend(model=self.get_local_model())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_key(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Replace the api_key value with masked placeholder."""
        if "api_key" in entry:
            entry["api_key"] = "***"
        return entry

    def to_dict(self) -> Dict[str, Any]:
        """Return the full config with keys masked (safe for API responses)."""
        return {
            "inference_mode": self.get_inference_mode(),
            "local_model": self.get_local_model(),
            "active_provider_id": self.get_active_provider_id(),
            "providers": self.list_providers(),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_registry: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    """Return the module-level ProviderRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry
