# cloud/ingestion/fusion/multimodal_fusion.py
# Stage A6 — Multimodal Fusion ★ MASTER CONTRACT.
#
# Aligns transcript segments with VLM captions and OCR results by
# nearest timestamp (tolerance ±2 seconds).
#
# Reads: transcript.json, vlm_output.jsonl (stream), ocr_output.jsonl (stream).
# Writes: cloud_runtime/lectures/{lecture_id}/multimodal_chunks.json
#
# Environment: Kaggle GPU only.
#
# CRITICAL: Output must validate against MultimodalChunk schema exactly.
# No extra fields. No renamed fields. No schema modifications.

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import CloudSettings
from schemas.chunk import MultimodalChunk

logger = logging.getLogger(__name__)

# Timestamp alignment tolerance in seconds.
ALIGNMENT_TOLERANCE = 2.0


def _stream_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Stream-read a JSONL file line by line.

    Returns:
        List of parsed dicts. Reads one line at a time to avoid
        loading the entire file into memory as a single JSON blob.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.error("JSONL parse error at %s:%d — %s", path, line_num, exc)
                raise
    return records


def _find_nearest(
    timestamp: float,
    records: List[Dict[str, Any]],
    key: str = "timestamp",
    frame_id_key: str = "frame_id",
) -> Optional[Dict[str, Any]]:
    """
    Find the record with the nearest timestamp within ±ALIGNMENT_TOLERANCE.

    Args:
        timestamp: Target timestamp in seconds.
        records: List of records with a timestamp-like field.
        key: The timestamp field name to compare against.
        frame_id_key: Not used for matching — just for logging.

    Returns:
        Nearest record dict, or None if nothing within tolerance.
    """
    best: Optional[Dict[str, Any]] = None
    best_dist = float("inf")

    for rec in records:
        rec_ts = rec.get(key, rec.get(frame_id_key, 0))
        # If the record uses frame_id and we mapped timestamps externally,
        # we need the caller to provide timestamp-mapped records.
        dist = abs(rec_ts - timestamp)
        if dist < best_dist:
            best_dist = dist
            best = rec

    if best is not None and best_dist <= ALIGNMENT_TOLERANCE:
        return best
    return None


def _build_frame_timestamp_index(
    vlm_records: List[Dict[str, Any]],
    ocr_records: List[Dict[str, Any]],
    frames_data: List[Dict[str, Any]],
) -> tuple:
    """
    Build timestamp-indexed lookups for VLM and OCR records.

    Since VLM/OCR records have frame_id but not timestamps directly,
    we map frame_id → timestamp from frames.json, then index by timestamp.

    Returns:
        (vlm_by_ts, ocr_by_ts) — dicts mapping timestamp → record.
    """
    # Build frame_id → timestamp map from frames.json.
    frame_ts_map: Dict[int, float] = {}
    for frame in frames_data:
        frame_ts_map[frame["frame_id"]] = frame["timestamp"]

    # Index VLM records by their frame's timestamp.
    vlm_ts_records: List[Dict[str, Any]] = []
    for rec in vlm_records:
        ts = frame_ts_map.get(rec["frame_id"])
        if ts is not None:
            vlm_ts_records.append({**rec, "_timestamp": ts})

    # Index OCR records by their frame's timestamp.
    ocr_ts_records: List[Dict[str, Any]] = []
    for rec in ocr_records:
        ts = frame_ts_map.get(rec["frame_id"])
        if ts is not None:
            ocr_ts_records.append({**rec, "_timestamp": ts})

    return vlm_ts_records, ocr_ts_records


def fuse(
    lecture_id: str,
    cloud_settings: CloudSettings | None = None,
) -> List[MultimodalChunk]:
    """
    Execute multimodal fusion: align transcript, VLM, and OCR by timestamp.

    Reads transcript.json, vlm_output.jsonl, and ocr_output.jsonl from
    the lecture's cloud_runtime directory.

    Args:
        lecture_id: Unique lecture identifier.
        cloud_settings: Injected CloudSettings.

    Returns:
        List of validated MultimodalChunk instances.

    Raises:
        FileNotFoundError: If required input files are missing.
        ValidationError: If any output chunk fails schema validation.
    """
    settings = cloud_settings or CloudSettings()
    lecture_dir = Path(settings.lecture_dir(lecture_id))

    # --- Load inputs ---
    transcript_path = lecture_dir / "transcript.json"
    vlm_path = lecture_dir / "vlm_output.jsonl"
    ocr_path = lecture_dir / "ocr_output.jsonl"
    frames_path = lecture_dir / "frames.json"

    if not transcript_path.exists():
        raise FileNotFoundError(f"A6: transcript.json not found: {transcript_path}")
    if not frames_path.exists():
        raise FileNotFoundError(f"A6: frames.json not found: {frames_path}")

    # Load transcript segments.
    transcript_segments = json.loads(transcript_path.read_text(encoding="utf-8"))

    # Load frames.json for timestamp mapping.
    frames_data = json.loads(frames_path.read_text(encoding="utf-8"))

    # Stream-read JSONL files.
    vlm_records = _stream_jsonl(vlm_path) if vlm_path.exists() else []
    ocr_records = _stream_jsonl(ocr_path) if ocr_path.exists() else []

    # Build timestamp-indexed lookups.
    vlm_ts_records, ocr_ts_records = _build_frame_timestamp_index(
        vlm_records, ocr_records, frames_data,
    )

    # --- Align and fuse ---
    chunks: List[MultimodalChunk] = []

    for chunk_num, seg in enumerate(transcript_segments, start=1):
        seg_timestamp = seg.get("start", 0.0)

        # Find nearest VLM caption.
        visual_context = ""
        nearest_vlm = _find_nearest(seg_timestamp, vlm_ts_records, key="_timestamp")
        if nearest_vlm is not None:
            visual_context = nearest_vlm.get("caption", "")

        # Find nearest OCR text.
        ocr_text = ""
        nearest_ocr = _find_nearest(seg_timestamp, ocr_ts_records, key="_timestamp")
        if nearest_ocr is not None:
            ocr_list = nearest_ocr.get("ocr_text", [])
            if isinstance(ocr_list, list):
                ocr_text = " ".join(ocr_list)
            else:
                ocr_text = str(ocr_list)

        # Build chunk_id: {lecture_id}_chunk_{chunk_number}
        chunk_id = f"{lecture_id}_chunk_{chunk_num:06d}"

        # Validate against MultimodalChunk schema — raises on failure.
        chunk = MultimodalChunk(
            lecture_id=lecture_id,
            chunk_id=chunk_id,
            timestamp=round(seg_timestamp, 3),
            transcript=seg.get("text", ""),
            visual_context=visual_context,
            ocr_text=ocr_text,
        )
        chunks.append(chunk)

    if not chunks:
        raise ValueError("A6: Fusion produced zero chunks.")

    # --- Write output ---
    output_path = lecture_dir / "multimodal_chunks.json"
    output_path.write_text(
        json.dumps([c.model_dump() for c in chunks], indent=2),
        encoding="utf-8",
    )
    logger.info(
        "A6: multimodal_chunks.json written (%d chunks) to %s",
        len(chunks), output_path,
    )

    return chunks
