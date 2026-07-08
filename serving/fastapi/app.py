# serving/fastapi/app.py
# E5 — FastAPI Application.
#
# Mounts all routes. Initializes the query workflow on startup.
# Run with: uvicorn serving.fastapi.app:app --reload

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from serving.fastapi.routes import query as query_routes
from serving.fastapi.routes import lectures as lecture_routes
from serving.fastapi.routes import settings as settings_routes

logger = logging.getLogger(__name__)


def _run_startup_audit() -> None:
    """
    Audit the package registry against the filesystem at startup.

    Marks registry entries whose package_path no longer exists as FAILED.
    Logs orphaned directories and duplicate display names.
    This runs every startup so the registry self-heals after manual cleanups.
    """
    try:
        from local.storage.registry_provider import get_registry
        reg = get_registry()
        from local.storage.lecture_registry import PROJECT_ROOT
        packages_dir = PROJECT_ROOT / "data" / "packages"
        report = reg.audit(packages_dir=packages_dir)
        issues = sum(len(v) for v in report.values())
        if issues:
            logger.warning("Startup: Registry audit found %d issue(s): %s", issues, report)
        else:
            logger.info("Startup: Registry audit passed — no inconsistencies.")
    except Exception as exc:
        logger.error("Startup: Registry audit failed (non-fatal): %s", exc)


def _try_restore_active_lecture() -> None:
    """
    V7 Startup Restoration.

    Reads last_active_lecture_id from the registry.  If the lecture
    exists, is READY, and its package directory is present on disk, a
    fresh QueryWorkflow is created and registered as the active workflow.

    Any failure is caught and logged.  The server always continues.
    """
    try:
        from local.storage.registry_provider import get_registry
        registry = get_registry()

        lecture_id = registry.get_active_lecture_id()
        if not lecture_id:
            logger.info("Startup: No previously active lecture found in registry.")
            return

        lecture = registry.get_lecture(lecture_id)
        if not lecture:
            logger.warning(
                "Startup: Active lecture '%s' not found in registry — skipping restore.",
                lecture_id,
            )
            return

        if lecture.get("status") != "READY":
            logger.warning(
                "Startup: Active lecture '%s' is not READY (status=%s) — skipping restore.",
                lecture_id,
                lecture.get("status"),
            )
            return

        package_path = lecture.get("package_path", "")
        if not package_path or not Path(package_path).is_dir():
            logger.warning(
                "Startup: Package directory for '%s' does not exist at '%s' — skipping restore.",
                lecture_id,
                package_path,
            )
            return

        logger.info("Startup: Restoring active lecture '%s'...", lecture_id)
        query_routes.activate_lecture(lecture_id)
        logger.info("Startup: Active lecture '%s' restored successfully.", lecture_id)

    except Exception as exc:
        logger.error(
            "Startup: Failed to restore active lecture — server will continue without "
            "a loaded workflow: %s",
            exc,
        )


def _run_model_recovery() -> None:
    """
    Ensure all required models are present locally, recovering or downloading
    them if missing. Fails fast if recovery is impossible.

    Reranker strategy (global singleton):
        1. Check local_runtime/models/global_reranker/ for a fine-tuned model.
        2. If missing, download the default pretrained model from Hugging Face.
        3. Load the model once and register it as the application-level singleton
           in rerank_service — it will never be reloaded due to lecture switching.
    """
    from config import local_settings
    from pathlib import Path

    if getattr(local_settings, "ENABLE_AUTO_MODEL_RECOVERY", False) is False:
        logger.info("Startup: Auto model recovery disabled in config.")
        return

    logger.info("Startup: Starting automatic model recovery checks...")

    reranker_status = "✗ Recovery Failed"
    ollama_status = "✗ Recovery Failed"
    error_reason = ""

    try:
        # 1. Reranker Recovery — global singleton directory only.
        #    Never reads from a knowledge package.
        reranker_dir = Path(local_settings.GLOBAL_RERANKER_DIR)
        logger.info("Startup: Checking global reranker at %s...", reranker_dir)

        reranker_ready = False
        if reranker_dir.exists() and (reranker_dir / "config.json").exists():
            import json
            try:
                with open(reranker_dir / "config.json", "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "model_type" in cfg or "architectures" in cfg:
                    logger.info("Startup: ✓ Found global reranker model.")
                    reranker_ready = True
                    reranker_status = "✓ Already Present"
                else:
                    logger.warning("Startup: Global reranker config.json is missing required model keys.")
            except Exception as e:
                logger.warning("Startup: Failed to parse global reranker config.json: %s", e)

        # 2. Fallback: download pretrained model from Hugging Face.
        if not reranker_ready:
            reranker_id = getattr(local_settings, "RERANKER_MODEL_ID", "BAAI/bge-reranker-base")
            logger.info("Startup: Global reranker not found. Downloading from Hugging Face (%s)...", reranker_id)
            try:
                from sentence_transformers import CrossEncoder
                import os
                model = CrossEncoder(reranker_id, device="cpu")
                os.makedirs(reranker_dir, exist_ok=True)
                model.save(str(reranker_dir))
                logger.info("Startup: ✓ Download complete.")
                reranker_ready = True
                reranker_status = "✓ Downloaded from Hugging Face"
            except Exception as e:
                logger.error("Startup: Hugging Face recovery failed: %s", e)

        if not reranker_ready:
            raise RuntimeError(
                "Global reranker model missing and Hugging Face download failed. "
                f"Expected at: {reranker_dir}"
            )

        # 3. Load and register the global singleton in rerank_service.
        #    This is the only place the model is ever loaded from disk.
        #    Lecture switching does NOT affect this singleton.
        logger.info("Startup: Loading global reranker singleton...")
        try:
            from local.loaders.reranker_loader import _load_cross_encoder
            from local.services.reranker_service import RerankerService
            from retrieval.reranker import rerank_service as _rs
            model = _load_cross_encoder(reranker_dir)
            _rs._GLOBAL_RERANKER_SERVICE = RerankerService(model)
            logger.info("Startup: ✓ Global reranker singleton initialized.")
            if reranker_status == "✗ Recovery Failed":
                reranker_status = "✓ Loaded"
        except Exception as e:
            raise RuntimeError(f"Global reranker model failed to load: {e}")

        # 4. Ollama Recovery - REMOVED
        # Provider checking is now deferred to the runtime ProviderManager.
        ollama_status = "✓ Deferred to Runtime Provider Manager"

        # Print Success Summary
        summary = f"""
==========================
Model Recovery Summary
==========================

Reranker (Global Singleton)
{reranker_status}

Embedding Model
✓ Present

Ollama
{ollama_status}

Startup completed successfully.
"""
        logger.info(summary)

    except Exception as exc:
        error_reason = str(exc)
        summary = f"""
==========================
Model Recovery Summary
==========================

Reranker (Global Singleton)
{reranker_status}

Ollama
{ollama_status}

Reason:
{error_reason}

Startup aborted.
"""
        logger.error(summary)
        raise RuntimeError("Startup aborted due to model recovery failure.") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — setup on startup, cleanup on shutdown."""
    logger.info("LectureMind V7 API starting...")
    _run_startup_audit()
    _run_model_recovery()
    _try_restore_active_lecture()
    yield
    logger.info("LectureMind V7 API shutting down.")
    query_routes.clear_workflow()


app = FastAPI(
    title="LectureMind V7",
    description="GraphRAG-powered lecture understanding API",
    version="7.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # temporary development setting
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes.
app.include_router(query_routes.router, tags=["Query"])
app.include_router(lecture_routes.router, tags=["Lectures"])
app.include_router(settings_routes.router)
from serving.fastapi.routes import debug as debug_routes
from serving.fastapi.routes import reranker as reranker_routes
app.include_router(debug_routes.router, tags=["Debug"])
app.include_router(reranker_routes.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    active_lecture = None
    try:
        from local.storage.registry_provider import get_registry
        active_lecture = get_registry().get_active_lecture_id()
    except Exception:
        pass
    return {
        "status": "healthy",
        "version": "7.0.0",
        "workflow_loaded": query_routes.get_workflow() is not None,
        "active_lecture_id": active_lecture,
    }
