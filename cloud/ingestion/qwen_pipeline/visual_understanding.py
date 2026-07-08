# cloud/ingestion/qwen_pipeline/visual_understanding.py
# Stage A4 — Diagram/Visual Understanding (Qwen2-VL-7B-Instruct).
#
# Processes key frames through Qwen2-VL to generate captions and objects.
# Output: cloud_runtime/lectures/{lecture_id}/vlm_output.jsonl (JSONL, streaming).
# Environment: Kaggle GPU only.
#
# Optimizations:
#   - Images resized to 768px max dimension before inference (reduces visual tokens).
#   - Batched generation (BATCH_SIZE=4) instead of one-at-a-time.
#   - max_new_tokens reduced from 256 to 96.
#   - Progress logging every 10 frames.
#
# Loaded in float16 via Transformers for T4/P100 VRAM constraints.
# Writes JSONL incrementally — never loads entire output into memory.

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from config import CloudSettings
from schemas.vlm import VLMCaption

logger = logging.getLogger(__name__)

# Prompt for Qwen2-VL visual understanding.
VLM_PROMPT = (
    "Describe this educational slide or diagram in detail. "
    "Identify all key objects, concepts, formulas, code snippets, or diagrams visible. "
    "Return a concise caption and list the main objects."
)

# Batch size for batched generation.
BATCH_SIZE = 4

# Maximum dimension for inference images (preserves aspect ratio).
MAX_INFERENCE_DIM = 768

# Maximum generated tokens per frame.
MAX_NEW_TOKENS = 96


def _load_qwen_model() -> tuple:
    """
    Load Qwen2-VL-7B-Instruct from local Kaggle path without quantization.

    Returns:
        (model, processor) tuple.
    """
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from config import CloudSettings
    from pathlib import Path

    settings = CloudSettings()
    
    logger.info(
        "A4: Loading Qwen model from %s",
        settings.QWEN_VL_MODEL_PATH
    )

    if not Path(settings.QWEN_VL_MODEL_PATH).exists():
        raise FileNotFoundError(f"Model path does not exist: {settings.QWEN_VL_MODEL_PATH}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        settings.QWEN_VL_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        settings.QWEN_VL_MODEL_PATH,
        trust_remote_code=True,
    )

    logger.info(
        "A4: Qwen2-VL loaded successfully without BitsAndBytes"
    )

    return model, processor


def _resize_for_inference(image: Any) -> Any:
    """
    Resize an image so its largest dimension is at most MAX_INFERENCE_DIM.

    Preserves aspect ratio. Does NOT modify the stored frame file on disk.
    Only used for inference input.
    """
    w, h = image.size
    if max(w, h) <= MAX_INFERENCE_DIM:
        return image
    image.thumbnail((MAX_INFERENCE_DIM, MAX_INFERENCE_DIM))
    return image


def _caption_single_frame(
    model: Any,
    processor: Any,
    image_path: Path,
    frame_id: int,
) -> VLMCaption:
    """
    Generate a caption for a single frame image.

    Returns:
        Validated VLMCaption instance.
    """
    from qwen_vl_utils import process_vision_info
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image = _resize_for_inference(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": VLM_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    # Trim the input tokens to get only the generated response.
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Parse caption and objects from the raw output.
    caption = raw_text if raw_text else "No caption generated"
    # Extract simple object list — split on commas/newlines from the output.
    objects = _extract_objects(raw_text)

    return VLMCaption(
        frame_id=frame_id,
        caption=caption,
        objects=objects,
    )


def _caption_batch(
    model: Any,
    processor: Any,
    batch_items: List[Dict[str, Any]],
) -> List[VLMCaption]:
    """
    Generate captions for a batch of frames in a single forward pass.

    Falls back to sequential processing if batched inference fails
    (e.g., due to variable image sizes causing tensor shape issues).

    Args:
        model: Qwen2-VL model instance.
        processor: Qwen2-VL processor instance.
        batch_items: List of dicts with 'frame_id' and 'image_path'.

    Returns:
        List of VLMCaption instances.
    """
    try:
        from qwen_vl_utils import process_vision_info
        from PIL import Image

        images = []
        messages_list = []
        for item in batch_items:
            img = Image.open(item["image_path"]).convert("RGB")
            img = _resize_for_inference(img)
            images.append(img)
            messages_list.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": VLM_PROMPT},
                    ],
                }
            ])

        texts = [
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_list
        ]

        # Collect all image inputs.
        all_image_inputs = []
        for msgs in messages_list:
            img_inputs, _ = process_vision_info(msgs)
            if img_inputs:
                all_image_inputs.extend(img_inputs)

        inputs = processor(
            text=texts,
            images=all_image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        results = []
        for i, item in enumerate(batch_items):
            # For batched output, each row corresponds to one input.
            gen_ids = output_ids[i, inputs.input_ids.shape[1]:]
            raw_text = processor.batch_decode([gen_ids], skip_special_tokens=True)[0].strip()
            caption = raw_text if raw_text else "No caption generated"
            objects = _extract_objects(raw_text)
            results.append(VLMCaption(
                frame_id=item["frame_id"],
                caption=caption,
                objects=objects,
            ))
        logger.info("A4: Batched inference active — %d frames processed in single forward pass.", len(results))
        return results

    except Exception as exc:
        # Fallback: process sequentially if batching fails.
        logger.warning("A4: Falling back to sequential inference. Batch error: %s", exc)
        results = []
        for item in batch_items:
            vlm_caption = _caption_single_frame(
                model, processor, item["image_path"], item["frame_id"],
            )
            results.append(vlm_caption)
        return results


def _extract_objects(raw_text: str) -> List[str]:
    """Extract object keywords from VLM output text."""
    # Simple heuristic: look for comma-separated items or keywords.
    import re
    # Try to find a list pattern in the text.
    tokens = re.split(r"[,\n;]", raw_text)
    objects = []
    for token in tokens:
        cleaned = token.strip().lower()
        # Keep short descriptive tokens as objects.
        if cleaned and len(cleaned) < 50 and len(cleaned) > 1:
            objects.append(cleaned)
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for obj in objects:
        if obj not in seen:
            seen.add(obj)
            unique.append(obj)
    return unique[:20]  # Cap at 20 objects per frame.


def _log_gpu_memory(label: str):
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total - reserved
            logger.info(
                f"[DIAGNOSTIC] {label} | "
                f"Allocated: {allocated:.2f}GB | "
                f"Reserved: {reserved:.2f}GB | "
                f"Total: {total:.2f}GB | "
                f"Free: {free:.2f}GB"
            )
    except Exception as e:
        logger.warning(f"[DIAGNOSTIC] Failed to log GPU memory: {e}")


def run_visual_understanding(
    lecture_id: str,
    frames: List[Dict[str, Any]],
    cloud_settings: CloudSettings | None = None,
    model_loader: Any = None,
) -> int:
    """
    Process all key frames through Qwen2-VL and write vlm_output.jsonl.

    Uses batched generation (BATCH_SIZE=4) with progress logging every 10 frames.

    Args:
        lecture_id: Unique lecture identifier.
        frames: List of frame dicts (from frames.json) with frame_id and image_path.
        cloud_settings: Injected CloudSettings.
        model_loader: Optional callable returning (model, processor) for testing.

    Returns:
        Number of frames processed.
    """
    settings = cloud_settings or CloudSettings()
    output_dir = Path(settings.lecture_dir(lecture_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vlm_output.jsonl"

    # Load model once — reuse for all frames.
    if model_loader is not None:
        model, processor = model_loader()
    else:
        model, processor = _load_qwen_model()

    lecture_dir = Path(settings.lecture_dir(lecture_id))
    total_frames = len(frames)
    count = 0

    logger.info("A4: Processing %d frames (batch_size=%d, max_tokens=%d)...",
                total_frames, BATCH_SIZE, MAX_NEW_TOKENS)

    # JSONL: append one record per line, never load entire file.
    with open(output_path, "w", encoding="utf-8") as f:
        # Process in batches.
        for batch_start in range(0, total_frames, BATCH_SIZE):
            batch_frames = frames[batch_start:batch_start + BATCH_SIZE]

            # Prepare batch items with resolved paths.
            batch_items = []
            for frame_data in batch_frames:
                frame_id = frame_data["frame_id"]
                image_path = lecture_dir / frame_data["image_path"]

                if not image_path.exists():
                    logger.warning("A4: Frame image not found: %s — skipping.", image_path)
                    continue

                batch_items.append({"frame_id": frame_id, "image_path": image_path})

            if not batch_items:
                continue

            try:
                captions = _caption_batch(model, processor, batch_items)
                for vlm_caption in captions:
                    f.write(json.dumps(vlm_caption.model_dump()) + "\n")
                    f.flush()
                    count += 1
            except Exception as exc:
                logger.error("A4: Failed to process batch starting at frame %d: %s",
                             batch_start, exc)
                raise

            # Progress logging every 10 frames.
            if count % 10 == 0 or count == total_frames:
                logger.info("A4: Processed %d/%d frames", count, total_frames)

    logger.info("A4: vlm_output.jsonl written (%d records) to %s", count, output_path)
    
    _log_gpu_memory("Immediately before cleanup")
    # GPU Memory Cleanup: Release Qwen2-VL model after A4 processing.
    try:
        del model
    except Exception:
        pass
    try:
        del processor
    except Exception:
        pass
    
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    
    _log_gpu_memory("Immediately after cleanup")
    _log_gpu_memory("Immediately before returning from run_visual_understanding()")
    return count
