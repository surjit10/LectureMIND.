# cloud/orchestration/llm_backend.py
# LLM Backend Abstraction for LectureMind V2.
#
# Provides a unified interface for text generation across all pipeline stages.
# Currently implements TransformersBackend only.
# VLLMBackend can be added in the future without changing business logic.

import gc
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseLLMBackend(ABC):
    """Abstract base class for LLM inference backends."""

    @abstractmethod
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text completions for a list of prompts.

        Args:
            prompts: List of input prompts.
            **kwargs: Backend-specific generation parameters
                      (temperature, max_tokens, top_p).

        Returns:
            List of generated text strings (one per prompt).
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release model resources and free GPU memory."""
        ...


class TransformersBackend(BaseLLMBackend):
    """
    Hugging Face Transformers backend for text generation.

    Uses AutoModelForCausalLM + AutoTokenizer for local model inference.
    Compatible with Kaggle's default PyTorch environment.
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 1024,
        torch_dtype: str = "float16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # VALIDATION: Check model architecture type.
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if getattr(config, "model_type", None) == "qwen2_vl":
            raise ValueError(
                "Vision model supplied to text backend. "
                "Use QWEN_TEXT_MODEL_PATH instead."
            )

        logger.info("TransformersBackend: Loading model from %s ...", model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, torch_dtype),
            device_map="auto",
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens
        logger.info("TransformersBackend: Model loaded successfully.")

    def generate(
        self,
        prompts: List[str],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[str]:
        """
        Generate text completions for a list of prompts.

        Processes prompts sequentially to stay within GPU memory limits.
        """
        import torch

        results: List[str] = []

        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3072,
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, self.max_new_tokens),
                    temperature=max(temperature, 0.01),
                    top_p=top_p,
                    do_sample=temperature > 0.01,
                    pad_token_id=self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id,
                )

            # Decode only the generated portion (exclude the input tokens).
            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(text.strip())

        return results

    def count_tokens(self, text: str) -> int:
        """Count tokens accurately using the loaded model tokenizer."""
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            raise RuntimeError("TransformersBackend tokenizer is not initialized.")
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def cleanup(self) -> None:
        """Release model and tokenizer, free GPU memory."""
        import torch

        logger.info("TransformersBackend: Cleaning up GPU memory...")
        try:
            del self.model
            del self.tokenizer
        except AttributeError:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        logger.info("TransformersBackend: GPU memory released.")


def load_shared_llm(
    model_path: Optional[str] = None,
    max_new_tokens: int = 1024,
) -> TransformersBackend:
    """
    Load the shared LLM backend for text generation stages (A7, A8, A9, B1).

    Uses local Kaggle model path by default.

    Called exactly once by run_ingestion_pipeline.py and injected
    into A7, A8, A9, B1 via their llm_loader parameters.

    Args:
        model_path: Model path (defaults to CloudSettings.QWEN_TEXT_MODEL_PATH).
        max_new_tokens: Maximum new tokens to generate.

    Returns:
        TransformersBackend instance.
    """
    from config import CloudSettings

    settings = CloudSettings()

    if model_path is None:
        model_path = settings.QWEN_TEXT_MODEL_PATH

    return TransformersBackend(
        model_path=model_path,
        max_new_tokens=max_new_tokens,
    )
