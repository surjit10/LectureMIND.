# serving/fastapi/routes/settings.py
# LLM configuration endpoints.
#
# Endpoints:
#   GET  /settings                    — current config (keys masked)
#   PATCH /settings                   — update inference_mode, local_model, active_provider_id
#   GET  /settings/providers          — list all configured providers (keys masked)
#   POST /settings/providers          — add a new online provider
#   PATCH /settings/providers/{id}    — update a provider
#   DELETE /settings/providers/{id}   — remove a provider
#   POST /settings/providers/{id}/test — test provider connectivity
#   GET  /settings/ollama/models      — list locally available Ollama models

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from local.llm.provider_registry import get_provider_registry
from local.llm.provider_manager import get_provider_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["Settings"])


# ------------------------------------------------------------------
# Pydantic request/response models
# ------------------------------------------------------------------

class SettingsGetResponse(BaseModel):
    inference_mode: str
    local_model: str
    active_provider_id: Optional[str]
    providers: List[Dict[str, Any]]


class SettingsPatchRequest(BaseModel):
    inference_mode: Optional[str] = None
    local_model: Optional[str] = None
    active_provider_id: Optional[str] = None
    # Legacy field — kept for backward compat with old PUT /settings
    ollama_model: Optional[str] = None
    top_k: Optional[int] = None


class ProviderAddRequest(BaseModel):
    provider: str
    model: str
    api_key: str
    display_name: Optional[str] = None
    base_url: Optional[str] = None


class ProviderUpdateRequest(BaseModel):
    model: Optional[str] = None
    api_key: Optional[str] = None
    display_name: Optional[str] = None
    base_url: Optional[str] = None


class ProviderResponse(BaseModel):
    id: str
    provider: str
    model: str
    api_key: str          # always "***" in responses
    display_name: str
    base_url: Optional[str]


class TestConnectionRequest(BaseModel):
    provider: str
    api_key: str
    base_url: Optional[str] = None


class TestConnectionResponse(BaseModel):
    success: bool
    message: str
    models: List[str] = []


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.get("", response_model=SettingsGetResponse)
async def get_settings():
    """Return current AI configuration (API keys masked)."""
    reg = get_provider_registry()
    return SettingsGetResponse(**reg.to_dict())


@router.patch("", response_model=SettingsGetResponse)
async def patch_settings(body: SettingsPatchRequest):
    """
    Update AI configuration.

    Accepts:
      - inference_mode:      "offline" | "online"
      - local_model:         Ollama model name
      - active_provider_id:  UUID of the provider to activate
      - ollama_model:        alias for local_model (legacy compat)
    """
    reg = get_provider_registry()
    errors = []

    mode = body.inference_mode
    if mode is not None:
        try:
            reg.set_inference_mode(mode)
        except ValueError as exc:
            errors.append(str(exc))

    local_model = body.local_model or body.ollama_model
    if local_model is not None:
        try:
            reg.set_local_model(local_model)
        except ValueError as exc:
            errors.append(str(exc))

    dump = body.model_dump(exclude_unset=True)
    if "active_provider_id" in dump:
        try:
            reg.set_active_provider_id(dump["active_provider_id"])
        except (ValueError, KeyError) as exc:
            errors.append(str(exc))

    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    return SettingsGetResponse(**reg.to_dict())


@router.get("/providers", response_model=List[ProviderResponse])
async def list_providers():
    """List all configured online providers (API keys masked)."""
    return get_provider_registry().list_providers()


@router.post("/providers", response_model=ProviderResponse, status_code=201)
async def add_provider(body: ProviderAddRequest):
    """Register a new online LLM provider."""
    _validate_provider(body.provider)
    try:
        entry = get_provider_registry().add_provider(
            provider=body.provider,
            model=body.model,
            api_key=body.api_key,
            display_name=body.display_name,
            base_url=body.base_url,
        )
        return entry
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.patch("/providers/{provider_id}", response_model=ProviderResponse)
async def update_provider(provider_id: str, body: ProviderUpdateRequest):
    """Update an existing provider (model, API key, display_name, base_url)."""
    try:
        entry = get_provider_registry().update_provider(
            provider_id=provider_id,
            model=body.model,
            api_key=body.api_key,
            display_name=body.display_name,
            base_url=body.base_url,
        )
        return entry
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/providers/{provider_id}", status_code=204)
async def delete_provider(provider_id: str):
    """Remove a configured provider."""
    try:
        get_provider_registry().delete_provider(provider_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/providers/{provider_id}/test", response_model=TestConnectionResponse)
async def test_existing_provider(provider_id: str):
    """Test connectivity for a configured provider and fetch models."""
    reg = get_provider_registry()
    entry = reg.get_provider_raw(provider_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found.")

    try:
        from local.llm.online_backend import OnlineBackend
        success, msg, models = OnlineBackend.test_connection(
            provider=entry["provider"],
            api_key=entry["api_key"],
            base_url=entry.get("base_url"),
        )
        return TestConnectionResponse(success=success, message=msg, models=models)
    except Exception as exc:
        logger.error("Provider test failed: %s", exc)
        return TestConnectionResponse(success=False, message=str(exc))

@router.post("/providers/test", response_model=TestConnectionResponse)
async def test_provider(body: TestConnectionRequest):
    """Test connectivity for an online provider and fetch available models."""
    _validate_provider(body.provider)
    try:
        from local.llm.online_backend import OnlineBackend
        success, msg, models = OnlineBackend.test_connection(
            provider=body.provider,
            api_key=body.api_key,
            base_url=body.base_url,
        )
        return TestConnectionResponse(success=success, message=msg, models=models)
    except Exception as exc:
        logger.error("Provider test failed: %s", exc)
        return TestConnectionResponse(success=False, message=str(exc))


@router.get("/ollama/models")
async def list_ollama_models():
    """List all locally available Ollama models."""
    try:
        from local.llm.ollama_backend import OllamaBackend
        backend = OllamaBackend()
        models = backend.list_local_models()
        return {"models": models}
    except Exception as exc:
        logger.error("Failed to list Ollama models: %s", exc)
        return {"models": [], "error": str(exc)}


class DownloadModelRequest(BaseModel):
    model_name: str

@router.post("/ollama/download")
async def download_ollama_model(body: DownloadModelRequest):
    """Explicitly trigger an Ollama model download at runtime."""
    mgr = get_provider_manager()
    success = mgr.download_offline_model(body.model_name)
    if success:
        return {"status": "success", "message": f"Model {body.model_name} downloaded successfully."}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to download model {body.model_name}.")

@router.get("/status")
async def get_active_provider_status():
    """Get the current operational status of the active provider."""
    mgr = get_provider_manager()
    return mgr.get_active_provider_status()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_VALID_PROVIDERS = {
    "google_gemini", "openai", "groq", "openrouter", "anthropic", "custom"
}


def _validate_provider(provider: str) -> None:
    if provider not in _VALID_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid provider '{provider}'. "
                f"Valid choices: {sorted(_VALID_PROVIDERS)}."
            ),
        )
