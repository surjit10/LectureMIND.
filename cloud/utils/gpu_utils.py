# cloud/utils/gpu_utils.py
# GPU Memory Diagnostics (Optional, for debugging only).
#
# Provides GPU memory monitoring utilities to track memory usage
# across pipeline stages without modifying pipeline behavior.
#
# Usage:
#   from cloud.utils.gpu_utils import log_gpu_memory
#   log_gpu_memory("Stage A7 - After embeddings")

import logging

logger = logging.getLogger(__name__)


def log_gpu_memory(stage_name: str) -> None:
    """
    Log current GPU memory usage (if CUDA available).
    
    This is a diagnostic utility only — does not affect pipeline behavior.
    
    Args:
        stage_name: Human-readable stage identifier for logging.
    """
    try:
        import torch
    except ImportError:
        logger.debug("PyTorch not available for GPU memory logging.")
        return
    
    if not torch.cuda.is_available():
        return
    
    try:
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        
        logger.info(
            "[GPU Memory] %s: %.2f GB allocated / %.2f GB reserved (Total: %.2f GB)",
            stage_name, allocated_gb, reserved_gb, total_gb,
        )
    except Exception as exc:
        logger.warning("[GPU Memory] Could not read GPU stats: %s", exc)
