# cloud/ingestion/metadata/metadata_extractor.py
# Stage A1 — Video Ingestion / Metadata.
#
# Extracts lecture_id, duration, and fps from a video file.
# Output: cloud_runtime/lectures/{lecture_id}/metadata.json
# Environment: Kaggle GPU only.

import json
import logging
from pathlib import Path

import cv2

from config import CloudSettings
from schemas.metadata import VideoMetadata

logger = logging.getLogger(__name__)


def extract_metadata(
    lecture_id: str,
    video_path: Path,
    cloud_settings: CloudSettings | None = None,
) -> VideoMetadata:
    """
    Extract video metadata and write metadata.json.

    Args:
        lecture_id: Unique identifier for this lecture.
        video_path: Absolute path to the source video file.
        cloud_settings: Injected CloudSettings (uses default if None).

    Returns:
        Validated VideoMetadata instance.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If video cannot be opened or metadata is invalid.
    """
    settings = cloud_settings or CloudSettings()
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps <= 0:
            raise ValueError(f"Invalid FPS value: {fps}")

        duration = frame_count / fps
    finally:
        cap.release()

    # Validate against Chunk 1 schema — raises ValidationError on failure.
    metadata = VideoMetadata(
        lecture_id=lecture_id,
        duration=duration,
        fps=fps,
    )

    # Write to cloud_runtime/lectures/{lecture_id}/metadata.json
    output_dir = Path(settings.lecture_dir(lecture_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metadata.json"

    output_path.write_text(
        json.dumps(metadata.model_dump(), indent=2),
        encoding="utf-8",
    )
    logger.info("A1: metadata.json written to %s", output_path)

    return metadata
