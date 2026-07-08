from fastapi import APIRouter
from local.llm.provider_registry import get_provider_registry

router = APIRouter()

@router.get("/debug_backend")
async def debug_backend():
    reg = get_provider_registry()
    backend = reg.get_active_backend()
    return {
        "backend_class": type(backend).__name__,
        "mode": reg.get_inference_mode(),
        "active_provider_id": reg.get_active_provider_id(),
        "providers_count": len(reg._cfg.get("providers", []))
    }
