# cloud/ingestion/paddleocr_pipeline/ocr_engine.py
# Stage A5 — OCR (Formulas, Code, Slide Text) via PaddleOCR 2.9.1.
#
# Extracts text from key frames using lang='en', use_angle_cls=True.
# Output: cloud_runtime/lectures/{lecture_id}/ocr_output.jsonl (JSONL, streaming).
# Environment: Kaggle GPU only.

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from config import CloudSettings
from schemas.ocr import OCRResult

logger = logging.getLogger(__name__)


def _init_ocr_engine(cloud_settings: CloudSettings | None = None) -> Any:
    """
    Initialize PaddleOCR engine.

    Defaults to CPU to prevent SIGABRT crashes on Kaggle when cuDNN
    libraries are missing.  PaddlePaddle's C++ runtime may call abort()
    during GPU initialization before Python exception handling can
    intercept — this kills the notebook process immediately.

    GPU is only used when CloudSettings.OCR_USE_GPU is explicitly True.
    """
    logger.info("[DIAGNOSTIC] Entering function")
    logger.info("[DIAGNOSTIC] CloudSettings creation")
    settings = cloud_settings or CloudSettings()
    
    logger.info("[DIAGNOSTIC] Reading OCR_USE_GPU")
    use_gpu = settings.OCR_USE_GPU

    logger.info("[DIAGNOSTIC] Beginning PaddleOCR import")
    from paddleocr import PaddleOCR
    logger.info("[DIAGNOSTIC] PaddleOCR import completed")

    if use_gpu:
        logger.info("A5: OCR_USE_GPU=True — attempting to initialize PaddleOCR on GPU.")
        try:
            logger.info("[DIAGNOSTIC] Beginning PaddleOCR constructor")
            ocr = PaddleOCR(
                lang="en",
                use_angle_cls=True,
                use_gpu=True,
                show_log=False,
            )
            logger.info("[DIAGNOSTIC] Constructor completed")
            # Force a trivial inference to surface deferred CUDA errors.
            import numpy as np
            _dummy = np.zeros((32, 32, 3), dtype=np.uint8)
            
            logger.info("[DIAGNOSTIC] Dummy inference begins (GPU path)")
            ocr.ocr(_dummy, cls=False)
            logger.info("[DIAGNOSTIC] Dummy inference completed")
            
            logger.info("A5: PaddleOCR engine initialized on GPU successfully.")
            logger.info("[DIAGNOSTIC] Return from function")
            return ocr
        except Exception as gpu_exc:
            logger.warning(
                "A5: GPU OCR initialization failed. Falling back to CPU. Error: %s", gpu_exc
            )
    else:
        logger.info("A5: OCR_USE_GPU=False — initializing PaddleOCR on CPU directly (safe default).")

    # CPU Initialization (either as default or fallback)
    logger.info("[DIAGNOSTIC] CPU initialization begins")
    logger.info("[DIAGNOSTIC] Beginning PaddleOCR constructor")
    ocr = PaddleOCR(
        lang="en",
        use_angle_cls=True,
        use_gpu=False,
        show_log=False,
    )
    logger.info("[DIAGNOSTIC] Constructor completed")
    logger.info("[DIAGNOSTIC] CPU initialization completed")
    logger.info("A5: PaddleOCR engine initialized on CPU (lang=en, angle_cls=True).")
    
    logger.info("[DIAGNOSTIC] Return from function")
    return ocr


def _extract_text_from_image(ocr_engine: Any, image_path: Path) -> List[str]:
    """
    Run OCR on a single image and return extracted text lines.

    Returns:
        List of detected text strings.
    """
    result = ocr_engine.ocr(str(image_path), cls=True)

    texts: List[str] = []
    if result and result[0]:
        for line in result[0]:
            # PaddleOCR returns [[box_coords], (text, confidence)].
            if line and len(line) >= 2:
                text_content = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                text_content = text_content.strip()
                if text_content:
                    texts.append(text_content)
    return texts


def run_ocr(
    lecture_id: str,
    frames: List[Dict[str, Any]],
    cloud_settings: CloudSettings | None = None,
    ocr_engine: Any = None,
) -> int:
    """
    Process all key frames through PaddleOCR and write ocr_output.jsonl.

    Args:
        lecture_id: Unique lecture identifier.
        frames: List of frame dicts (from frames.json) with frame_id and image_path.
        cloud_settings: Injected CloudSettings.
        ocr_engine: Optional pre-initialized OCR engine for testing.

    Returns:
        Number of frames processed.
    """
    settings = cloud_settings or CloudSettings()
    output_dir = Path(settings.lecture_dir(lecture_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ocr_output.jsonl"

    if ocr_engine is None:
        logger.info("[DIAGNOSTIC] Entering _init_ocr_engine()")
        ocr_engine = _init_ocr_engine(cloud_settings=settings)

    lecture_dir = Path(settings.lecture_dir(lecture_id))
    count = 0

    logger.info("[DIAGNOSTIC] Beginning OCR processing")
    # JSONL: write one record per line, append incrementally.
    with open(output_path, "w", encoding="utf-8") as f:
        for frame_data in frames:
            frame_id = frame_data["frame_id"]
            image_path = lecture_dir / frame_data["image_path"]

            if not image_path.exists():
                logger.warning("A5: Frame image not found: %s — skipping.", image_path)
                continue

            try:
                ocr_texts = _extract_text_from_image(ocr_engine, image_path)

                # Validate against Chunk 1 OCRResult schema.
                ocr_result = OCRResult(
                    frame_id=frame_id,
                    ocr_text=ocr_texts,
                )
                f.write(json.dumps(ocr_result.model_dump()) + "\n")
                f.flush()
                count += 1
                logger.debug("A5: OCR processed frame %d (%d text lines)", frame_id, len(ocr_texts))
            except Exception as exc:
                logger.error("A5: Failed to OCR frame %d: %s", frame_id, exc)
                raise

    logger.info("A5: ocr_output.jsonl written (%d records) to %s", count, output_path)
    return count
