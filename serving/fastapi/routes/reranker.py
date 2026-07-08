# serving/fastapi/routes/reranker.py
import logging
import shutil
import zipfile
import uuid
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from config import local_settings
from retrieval.reranker import rerank_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reranker", tags=["Reranker"])


class RerankerStatusResponse(BaseModel):
    status: str
    model_name: str
    custom_model: bool
    loaded: bool
    disk_size_mb: float
    last_updated: str
    max_upload_size_mb: float
    backend: str


class RerankerSettingsPatch(BaseModel):
    max_upload_size_mb: int

def _get_dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
    return total / (1024 * 1024)


def _safe_extract(zip_path: Path, extract_dir: Path):
    """Extract zip safely, mitigating ZipSlip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            member_path = (extract_dir / member).resolve()
            if not str(member_path).startswith(str(extract_dir.resolve())):
                raise ValueError("ZipSlip detected: Invalid path in zip file.")
        zf.extractall(extract_dir)


@router.post("/upload")
async def upload_global_reranker(file: UploadFile = File(...)):
    """Upload, validate, and atomically replace the global reranker."""
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported.")

    temp_dir = Path(local_settings.UPLOAD_TEMP_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    upload_id = uuid.uuid4().hex
    zip_path = temp_dir / f"reranker_{upload_id}.zip"
    extract_dir = temp_dir / f"extracted_{upload_id}"

    try:
        # Save ZIP
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Check size limit
        if zip_path.stat().st_size > local_settings.MAX_MODEL_UPLOAD_MB * 1024 * 1024:
            raise ValueError(f"Model exceeds maximum allowed size ({local_settings.MAX_MODEL_UPLOAD_MB}MB).")

        # Extract
        _safe_extract(zip_path, extract_dir)
        
        # Flatten directory if nested (e.g. if zip contains a single folder)
        contents = list(extract_dir.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            model_dir = contents[0]
        else:
            model_dir = extract_dir

        # Validate
        from local.loaders.reranker_loader import validate_reranker_directory
        validate_reranker_directory(model_dir)

        # Atomic Replacement
        global_dir = Path(local_settings.GLOBAL_RERANKER_DIR)
        backup_dir = global_dir.with_name(f"{global_dir.name}_backup_{upload_id}")
        
        if global_dir.exists():
            global_dir.rename(backup_dir)
            
        try:
            shutil.copytree(model_dir, global_dir)
        except Exception as e:
            if backup_dir.exists():
                if global_dir.exists():
                    shutil.rmtree(global_dir)
                backup_dir.rename(global_dir)
            raise RuntimeError(f"Atomic replacement failed: {e}")

        # Reload
        try:
            rerank_service.reload_global_reranker(str(global_dir))
        except Exception as e:
            # Rollback
            if global_dir.exists():
                shutil.rmtree(global_dir)
            if backup_dir.exists():
                backup_dir.rename(global_dir)
                # Try reloading backup
                try:
                    rerank_service.reload_global_reranker(str(global_dir))
                except Exception:
                    pass
            raise RuntimeError(f"Failed to reload model: {e}")
            
        # Cleanup Backup
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        return {"status": "success", "message": "Global reranker replaced successfully."}

    except Exception as e:
        logger.error(f"Reranker upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if zip_path.exists():
            zip_path.unlink()
        if extract_dir.exists():
            shutil.rmtree(extract_dir)


@router.get("/status", response_model=RerankerStatusResponse)
async def get_reranker_status():
    global_dir = Path(local_settings.GLOBAL_RERANKER_DIR)
    max_size = getattr(local_settings, "MAX_MODEL_UPLOAD_MB", 2000)
    
    if not global_dir.exists() or rerank_service._GLOBAL_RERANKER_SERVICE is None:
        return RerankerStatusResponse(
            status="Not Loaded",
            model_name="None",
            custom_model=False,
            loaded=False,
            disk_size_mb=0.0,
            last_updated="N/A",
            max_upload_size_mb=max_size,
            backend="CrossEncoder"
        )
    
    size_mb = _get_dir_size_mb(global_dir)
    is_custom = (global_dir / "config.json").exists()
    
    import datetime
    last_updated_ts = global_dir.stat().st_mtime
    last_updated_str = datetime.datetime.fromtimestamp(last_updated_ts).strftime('%B %d, %Y %I:%M %p')
    
    return RerankerStatusResponse(
        status="Loaded",
        model_name="Custom Global Reranker" if is_custom else local_settings.RERANKER_MODEL_ID,
        custom_model=is_custom,
        loaded=True,
        disk_size_mb=round(size_mb, 2),
        last_updated=last_updated_str,
        max_upload_size_mb=max_size,
        backend="CrossEncoder"
    )


@router.post("/reload")
async def reload_reranker():
    global_dir = Path(local_settings.GLOBAL_RERANKER_DIR)
    try:
        rerank_service.reload_global_reranker(str(global_dir))
        return {"status": "success", "message": "Model reloaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


@router.delete("/")
async def delete_reranker():
    """Delete the custom model and restore the default."""
    global_dir = Path(local_settings.GLOBAL_RERANKER_DIR)
    if global_dir.exists():
        shutil.rmtree(global_dir)
    
    # Run recovery to download HF default and set singleton
    from serving.fastapi.app import _run_model_recovery
    try:
        _run_model_recovery()
        return {"status": "success", "message": "Custom model deleted. Default model restored."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore default model: {e}")


@router.patch("/settings")
async def patch_reranker_settings(body: RerankerSettingsPatch):
    """
    Update MAX_MODEL_UPLOAD_MB at runtime.
    Persists the new value to .env so it survives server restarts.
    """
    if body.max_upload_size_mb < 1 or body.max_upload_size_mb > 100_000:
        raise HTTPException(status_code=400, detail="max_upload_size_mb must be between 1 and 100000.")

    # Update in-memory singleton immediately.
    local_settings.MAX_MODEL_UPLOAD_MB = body.max_upload_size_mb

    # Persist to .env so the value survives restarts.
    env_path = Path(".env")
    key = "MAX_MODEL_UPLOAD_MB"
    new_line = f"{key}={body.max_upload_size_mb}\n"

    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
        replaced = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}=") or line.strip().startswith(f"{key} ="):
                lines[i] = new_line
                replaced = True
                break
        if not replaced:
            lines.append(new_line)
        env_path.write_text("".join(lines), encoding="utf-8")
    else:
        env_path.write_text(new_line, encoding="utf-8")

    logger.info("Reranker settings updated: MAX_MODEL_UPLOAD_MB=%d", body.max_upload_size_mb)
    return {"status": "success", "max_upload_size_mb": body.max_upload_size_mb}
