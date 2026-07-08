# local/loaders/reranker_loader.py
# D3 — Reranker Model Loader.
#
# Loads the global CrossEncoder from local_runtime/models/global_reranker/.
# Called exactly once by serving/fastapi/app.py at application startup.
# Does NOT copy, move, or delete any files.
# Does NOT retrain, modify weights, or fine-tune.
#
# Environment: Local only. Never imports from cloud/.

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RerankerLoadError(Exception):
    """Raised when reranker loading fails."""
    pass


# _copy_model_to_local() has been removed.
# Knowledge packages no longer contain a reranker_model/ directory.
# The global reranker lives at local_runtime/models/global_reranker/
# and is managed exclusively by app.py startup logic.


def validate_reranker_directory(model_path: Path) -> None:
    """Validate that the given directory contains a valid HuggingFace model."""
    import json
    from config import local_settings
    
    if not model_path.exists() or not model_path.is_dir():
        raise ValueError(f"Directory {model_path} does not exist.")
        
    required_files = ["config.json", "tokenizer.json"]
    for rf in required_files:
        if not (model_path / rf).exists():
            raise ValueError(f"Missing required file: {rf}")
            
    if not (model_path / "model.safetensors").exists() and not (model_path / "pytorch_model.bin").exists():
        raise ValueError("Missing model weights file (model.safetensors or pytorch_model.bin).")
        
    try:
        with open(model_path / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "model_type" not in cfg and "architectures" not in cfg:
            raise ValueError("Invalid config.json: Missing model_type or architectures.")
    except Exception as e:
        raise ValueError(f"Failed to parse config.json: {e}")


def _load_cross_encoder(model_path: Path) -> Any:
    """Load CrossEncoder from the global model directory."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        str(model_path),
        device="cpu"
    )
    logger.info("D3: CrossEncoder loaded successfully.")
    return model
