# cloud/ingestion/whisper_pipeline/transcriber.py
# Stage A2 — Speech Transcription.
#
# Uses faster-whisper (CTranslate2) with model large-v3.
# GPU if available, falls back to CPU.
# Output: cloud_runtime/lectures/{lecture_id}/transcript.json
# Environment: Kaggle GPU only.
#
# Field names are FIXED — Stage A6 reads segment_id/start/end/text exactly.

import json
import logging
from pathlib import Path
from typing import Any, List

from config import CloudSettings
from schemas.transcript import TranscriptSegment

logger = logging.getLogger(__name__)


def _load_whisper_model(model_size: str) -> Any:
    """
    Load WhisperModel from local Kaggle dataset with GPU preference.
    Falls back to CPU if CUDA initialization fails.
    """

    from faster_whisper import WhisperModel

    settings = CloudSettings()

    model_path = settings.WHISPER_MODEL_PATH

    logger.info(
        "A2: Whisper path = %s",
        model_path
    )

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Whisper model path not found: {model_path}"
        )

    logger.info(
        "A2: Loading Whisper model from %s",
        model_path
    )

    try:
        model = WhisperModel(
            model_path,
            device="cuda",
            compute_type="float16",
        )

        logger.info(
            "A2: Whisper model loaded on CUDA (float16)"
        )

    except Exception as e:

        logger.warning(
            "A2: CUDA load failed (%s). Falling back to CPU.",
            str(e)
        )

        model = WhisperModel(
            model_path,
            device="cpu",
            compute_type="int8",
        )

        logger.info(
            "A2: Whisper model loaded on CPU (int8)"
        )

    return model


def transcribe(
    lecture_id: str,
    video_path: Path,
    cloud_settings: CloudSettings | None = None,
    model_size: str = "large-v3",
    model_loader: Any = None,
) -> List[TranscriptSegment]:
    """
    Transcribe a lecture video using faster-whisper.

    Args:
        lecture_id: Unique lecture identifier.
        video_path: Path to the source video file.
        cloud_settings: Injected CloudSettings.
        model_size: Whisper model size (default: large-v3).
        model_loader: Optional callable returning a WhisperModel for testing.

    Returns:
        List of validated TranscriptSegment instances.
    """
    settings = cloud_settings or CloudSettings()
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Load model — allow injection for testing.
    if model_loader is not None:
        model = model_loader()
    else:
        model = _load_whisper_model(model_size)

    # Transcribe — returns (segments_generator, info).
    segments_gen, info = model.transcribe(str(video_path), beam_size=5)
    logger.info("A2: Detected language '%s' (prob=%.2f)", info.language, info.language_probability)

    # Collect and validate each segment.
    results: List[TranscriptSegment] = []
    for idx, seg in enumerate(segments_gen, start=1):
        transcript_seg = TranscriptSegment(
            segment_id=idx,
            start=round(seg.start, 3),
            end=round(seg.end, 3),
            text=seg.text.strip(),
        )
        results.append(transcript_seg)

    if not results:
        raise ValueError("A2: Whisper produced zero transcript segments.")

    # Write transcript.json
    output_dir = Path(settings.lecture_dir(lecture_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "transcript.json"

    output_path.write_text(
        json.dumps([seg.model_dump() for seg in results], indent=2),
        encoding="utf-8",
    )
    logger.info("A2: transcript.json written (%d segments) to %s", len(results), output_path)

    # GPU Memory Cleanup: Release Whisper model after A2 transcription.
    try:
        del model
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

    return results
