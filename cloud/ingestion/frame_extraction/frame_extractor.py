# cloud/ingestion/frame_extraction/frame_extractor.py
# Stage A3 — Frame Extraction via slide-change detection.
#
# Uses Structural Similarity Index (SSIM) to detect significant
# slide changes instead of fixed 1 FPS sampling.
# Target: 80–95% frame reduction on typical slide-based lectures.
#
# Optimizations:
#   - 1 FPS sampling via frame stepping (skip 29 of every 30 frames).
#   - SSIM threshold lowered to 0.75 for fewer false positives.
#   - 2-second cooldown between keyframes to prevent burst captures.
#
# Output: cloud_runtime/lectures/{lecture_id}/frames.json + frames/
# Environment: Kaggle GPU only.

import json
import logging
from pathlib import Path
from typing import List

try:
    import cv2
except ImportError:
    # cv2 is a cloud-only dependency (Kaggle). Importing it at module level
    # made the module unimportable in local/CI environments, which broke the
    # test mock target resolution. Tests patch this module attribute.
    cv2 = None  # type: ignore[assignment]
import numpy as np

from config import CloudSettings
from schemas.frames import Frame

logger = logging.getLogger(__name__)

# Default SSIM threshold — frames with SSIM below this vs. previous
# key frame are considered a slide change.
DEFAULT_SSIM_THRESHOLD = 0.75

# Minimum time between keyframes in seconds.
DEFAULT_COOLDOWN_SECONDS = 2.0

# Maximum time between keyframes in seconds. When this interval elapses
# without an SSIM-detected change, a frame is force-captured as a safety
# keyframe so long static slides and subtle transitions that SSIM misses
# still receive visual coverage (A4 VLM / A5 OCR / A6 fusion).
DEFAULT_SAFETY_KEYFRAME_INTERVAL = 30.0


def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index between two grayscale images.

    Simplified SSIM (no gaussian weighting) for speed on Kaggle.
    Uses the standard SSIM formula:
        SSIM = (2*mu_x*mu_y + C1)(2*sigma_xy + C2) /
               ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(numerator / denominator)


def extract_frames(
    lecture_id: str,
    video_path: Path,
    cloud_settings: CloudSettings | None = None,
    ssim_threshold: float = DEFAULT_SSIM_THRESHOLD,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    safety_interval: float = DEFAULT_SAFETY_KEYFRAME_INTERVAL,
) -> List[Frame]:
    """
    Extract key frames from a lecture video using slide-change detection.

    Uses 1 FPS sampling to avoid scanning all video frames, a lowered SSIM
    threshold (0.75) to reduce false keyframes, a cooldown period to prevent
    burst captures from animations, and a periodic safety keyframe so a long
    stretch without an SSIM-detected change (e.g. a lecturer speaking on one
    static slide for minutes) still yields a frame for visual understanding.

    Args:
        lecture_id: Unique lecture identifier.
        video_path: Path to the source video file.
        cloud_settings: Injected CloudSettings.
        ssim_threshold: SSIM threshold below which a frame is a new slide.
        cooldown_seconds: Minimum seconds between consecutive keyframes.
        safety_interval: Maximum seconds allowed without a keyframe before
            a frame is force-captured. Must be >= cooldown_seconds (the
            safety check only runs once the cooldown gate has passed); the
            defaults satisfy this (30.0 >= 2.0).

    Returns:
        List of validated Frame instances.
    """
    settings = cloud_settings or CloudSettings()
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if cv2 is None:
        raise RuntimeError(
            "cv2 (opencv-python) is required for frame extraction but is not installed."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS: {fps}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample at 1 FPS — skip intervening frames entirely.
    frame_step = max(1, int(round(fps)))

    # Prepare output directories.
    output_dir = Path(settings.lecture_dir(lecture_id))
    frames_dir = Path(settings.frame_dir(lecture_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    key_frames: List[Frame] = []
    prev_gray: np.ndarray | None = None
    key_frame_count = 0
    sampled_count = 0
    total_read = 0
    last_keyframe_timestamp = -cooldown_seconds  # Allow first frame immediately.

    # Use seek-based reading if frame count is known; otherwise sequential.
    use_seek = total_video_frames > 0

    frame_idx = 0
    while True:
        if use_seek:
            if frame_idx >= total_video_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret:
            break
        total_read += 1

        # In sequential mode, skip frames that are not on the 1-FPS boundary.
        if not use_seek and (total_read - 1) % frame_step != 0:
            frame_idx = total_read  # Track position for timestamp calc.
            continue

        if not use_seek:
            frame_idx = total_read - 1

        sampled_count += 1
        timestamp = frame_idx / fps

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize for SSIM speed — 320px width is sufficient for similarity.
        h, w = gray.shape
        if w > 320:
            scale = 320 / w
            gray_resized = cv2.resize(gray, (320, int(h * scale)))
        else:
            gray_resized = gray

        is_key_frame = False
        if prev_gray is None:
            # Always capture the first frame.
            is_key_frame = True
        else:
            # Enforce cooldown — skip if too close to previous keyframe.
            if (timestamp - last_keyframe_timestamp) >= cooldown_seconds:
                ssim_val = _compute_ssim(prev_gray, gray_resized)
                if ssim_val < ssim_threshold:
                    is_key_frame = True
                elif (timestamp - last_keyframe_timestamp) >= safety_interval:
                    # Periodic safety keyframe: no SSIM change for a long
                    # stretch — force a capture so long static slides and
                    # subtle transitions are not lost entirely.
                    logger.info(
                        "A3: Safety keyframe at t=%.1fs (%.1fs since last capture).",
                        timestamp, timestamp - last_keyframe_timestamp,
                    )
                    is_key_frame = True

        if is_key_frame:
            key_frame_count += 1
            image_filename = f"frame_{key_frame_count:06d}.jpg"
            image_path = frames_dir / image_filename

            cv2.imwrite(str(image_path), frame)

            # Validate against Chunk 1 Frame schema.
            frame_record = Frame(
                frame_id=key_frame_count,
                timestamp=round(timestamp, 3),
                image_path=f"frames/{image_filename}",
            )
            key_frames.append(frame_record)
            prev_gray = gray_resized
            last_keyframe_timestamp = timestamp

        if use_seek:
            frame_idx += frame_step

    cap.release()

    if not key_frames:
        raise ValueError("A3: No key frames extracted from video.")

    reduction_pct = (1 - len(key_frames) / max(total_video_frames, 1)) * 100
    logger.info(
        "A3: Extracted %d key frames from %d total (%d sampled at 1 FPS, %.1f%% reduction)",
        len(key_frames), total_video_frames, sampled_count, reduction_pct,
    )

    # Write frames.json
    output_path = output_dir / "frames.json"
    output_path.write_text(
        json.dumps([f.model_dump() for f in key_frames], indent=2),
        encoding="utf-8",
    )
    logger.info("A3: frames.json written to %s", output_path)

    return key_frames
