# local/llm/provider_manager.py
# Runtime Provider Manager.
# Single source of truth for AI provider availability, status, and actions.
# Avoids startup side-effects by checking health only on explicit user request.

import logging
from typing import Dict, Any, List

from local.llm.provider_registry import get_provider_registry

logger = logging.getLogger(__name__)

class ProviderStatus:
    AVAILABLE = "Available"
    UNAVAILABLE = "Unavailable"
    MISSING_API_KEY = "Missing API Key"
    INVALID_API_KEY = "Invalid API Key"
    NETWORK_ERROR = "Network Error"
    MODEL_MISSING = "Model Missing"
    DOWNLOADING = "Downloading"
    DISABLED = "Disabled"
    NOT_INSTALLED = "Not Installed"
    NOT_CONFIGURED = "Not Configured"

class ProviderManager:
    """
    Manages operational status of LLM providers.
    Uses ProviderRegistry for the underlying storage.
    """
    def __init__(self):
        self.registry = get_provider_registry()

    def get_active_provider_status(self) -> Dict[str, Any]:
        """Get the status of the currently active provider based on inference_mode."""
        mode = self.registry.get_inference_mode()
        if mode == "online":
            pid = self.registry.get_active_provider_id()
            if not pid:
                return {"status": ProviderStatus.NOT_CONFIGURED, "message": "No online provider active."}
            entry = self.registry.get_provider_raw(pid)
            if not entry:
                return {"status": ProviderStatus.NOT_CONFIGURED, "message": "Active provider missing."}
            return self.check_online_status(entry)
        else:
            return self.check_offline_status(self.registry.get_local_model())

    def check_online_status(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        from local.llm.online_backend import OnlineBackend
        if not entry.get("api_key") or entry.get("api_key") == "***":
            return {"status": ProviderStatus.MISSING_API_KEY, "message": "API key is missing."}

        try:
            success, msg, _ = OnlineBackend.test_connection(
                provider=entry["provider"],
                api_key=entry["api_key"],
                base_url=entry.get("base_url")
            )
            if success:
                return {"status": ProviderStatus.AVAILABLE, "message": "Online provider is reachable."}
            elif "Invalid API key" in msg or "authentication" in msg.lower():
                return {"status": ProviderStatus.INVALID_API_KEY, "message": msg}
            else:
                return {"status": ProviderStatus.NETWORK_ERROR, "message": msg}
        except Exception as e:
            return {"status": ProviderStatus.UNAVAILABLE, "message": str(e)}

    def check_offline_status(self, model_name: str) -> Dict[str, Any]:
        try:
            from local.loaders.ollama_loader import check_model_ready
            # Pass model_name explicitly to avoid relying on config.py default at runtime
            is_ready = check_model_ready(ollama_client=None, model_name=model_name)
            if is_ready:
                return {"status": ProviderStatus.AVAILABLE, "message": f"Model {model_name} is ready."}
            else:
                return {"status": ProviderStatus.MODEL_MISSING, "message": f"Model {model_name} is not found locally."}
        except Exception as e:
            err_str = str(e).lower()
            if "connection" in err_str or "connect" in err_str:
                return {"status": ProviderStatus.NOT_INSTALLED, "message": "Ollama is not running or not installed."}
            return {"status": ProviderStatus.UNAVAILABLE, "message": str(e)}

    def download_offline_model(self, model_name: str) -> bool:
        """Explicitly trigger model download. Returns True on success."""
        from local.loaders.ollama_loader import pull_model_if_needed
        logger.info(f"ProviderManager: Triggering download for {model_name}")
        return pull_model_if_needed(model_name=model_name)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_manager = None

def get_provider_manager() -> ProviderManager:
    global _manager
    if _manager is None:
        _manager = ProviderManager()
    return _manager
