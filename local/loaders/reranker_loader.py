# local/loaders/reranker_loader.py
# D3 — Reranker Model Loader.
#
# Loads the global CrossEncoder from local_runtime/models/global_reranker/.
# Called exactly once by serving/fastapi/app.py at application startup.
# Does NOT copy, move, or delete any files.
# Does NOT retrain, modify weights, or fine-tune.
#
# V2 int8 quantization: when LocalSettings.RERANKER_QUANTIZE is enabled, the
# loaded cross-encoder is dynamically quantized to int8 (weights only). This
# is a pure in-memory optimization — the FP32 model on disk is never modified,
# and any quantization failure falls back to FP32 automatically.
#
# Environment: Local only. Never imports from cloud/.

import logging
from pathlib import Path
from typing import Any, Optional

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


def _apply_int8_quantization(model: Any) -> Any:
    """
    Dynamically quantize the cross-encoder transformer weights to int8.

    Locates the underlying transformers model (the CrossEncoder in
    sentence-transformers 5.x is a Sequential whose first module is the
    Transformer wrapper holding ``.auto_model``) and applies PyTorch's
    dynamic quantization to every Linear layer. Activations stay FP32,
    so this is safe on CPU and preserves ranking order.

    Args:
        model: A loaded CrossEncoder instance.

    Returns:
        The same CrossEncoder with the transformer swapped for the
        quantized copy, tagged ``model._precision = "int8_dynamic"``.

    Raises:
        RerankerLoadError: If the transformer cannot be located.
    """
    import torch

    # Locate the transformer module holding the nn.Linear weights.
    transformer = None
    parent = None
    attr = None

    # ST 5.x: CrossEncoder is a Sequential — ce[0] is the Transformer.
    if hasattr(model, "__getitem__") and len(model) > 0:
        parent = model[0]
        if hasattr(parent, "auto_model"):
            transformer, attr = parent.auto_model, "auto_model"
        else:
            transformer, attr = parent, None
    # Older ST / direct model attribute layout.
    elif hasattr(model, "model"):
        parent = model.model
        if hasattr(parent, "auto_model"):
            transformer, attr = parent.auto_model, "auto_model"
        else:
            transformer, attr = parent, None

    if transformer is None:
        raise RerankerLoadError(
            "Could not locate transformer for int8 quantization."
        )

    quantized = torch.quantization.quantize_dynamic(
        transformer,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    # Re-attach the quantized copy in place of the FP32 transformer.
    if attr is not None:
        setattr(parent, attr, quantized)
    elif hasattr(model, "__getitem__") and len(model) > 0:
        model[0] = quantized
    else:
        model.model = quantized

    model._precision = "int8_dynamic"
    logger.info("D3: CrossEncoder dynamically quantized to int8.")
    return model


def _load_cross_encoder(model_path: Path, quantize: Optional[bool] = None) -> Any:
    """
    Load CrossEncoder from the global model directory.

    Args:
        model_path: Directory containing the HF model files.
        quantize: When True, apply dynamic int8 quantization after loading.
            When None (default), reads LocalSettings.RERANKER_QUANTIZE.

    Returns:
        Loaded CrossEncoder. If quantization is requested but fails for any
        reason, the FP32 model is returned (never raises due to quantization).
    """
    from sentence_transformers import CrossEncoder
    from config import local_settings

    model = CrossEncoder(
        str(model_path),
        device="cpu"
    )

    if quantize is None:
        quantize = getattr(local_settings, "RERANKER_QUANTIZE", False)

    if quantize:
        try:
            model = _apply_int8_quantization(model)
        except Exception as exc:
            logger.warning(
                "D3: int8 quantization failed, falling back to FP32: %s", exc,
            )
            model._precision = "fp32"

    if not hasattr(model, "_precision"):
        model._precision = "fp32"

    logger.info("D3: CrossEncoder loaded successfully (precision=%s).", model._precision)
    return model
