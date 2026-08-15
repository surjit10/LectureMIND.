# cloud/ingestion/fusion/multimodal_fusion.py
# Stage A6 — Multimodal Fusion — MASTER CONTRACT.
#
# Aligns transcript segments with VLM captions and OCR results by matching
# frames against the FULL segment interval [start − tolerance, end + tolerance]
# (tolerance ±2 seconds), preferring the frame nearest the segment midpoint.
#
# V2 semantic chunk merging (opt-in via merge_segments=True): adjacent Whisper
# segments are merged into larger semantic chunks via a deterministic
# rolling-window algorithm (sentence-boundary-aware, silence-gap-aware).
#
# Reads: transcript.json, vlm_output.jsonl (stream), ocr_output.jsonl (stream).
# Writes: cloud_runtime/lectures/{lecture_id}/multimodal_chunks.json
#
# Environment: Kaggle GPU only.

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import CloudSettings
from schemas.chunk import MultimodalChunk

logger = logging.getLogger(__name__)

# Timestamp alignment tolerance in seconds.
ALIGNMENT_TOLERANCE = 2.0

# ── Semantic chunk merging constants ────────────────────────────────────
# These control the rolling-window merge in _merge_atoms().
# Values are tuned for lecture transcript text (~4 chars / token).
MIN_CHUNK_CHARS = 300          # Minimum merged-chunk length in characters.
TARGET_CHUNK_CHARS = 700       # Target — break at sentence boundaries near this.
MAX_CHUNK_CHARS = 1200         # Hard cap — always break here.
SILENCE_GAP_THRESHOLD = 1.5    # Seconds — gap larger than this is a topic break.


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


def _find_nearest_in_interval(
    start: float,
    end: float,
    records: List[Dict[str, Any]],
    key: str = "_timestamp",
    tolerance: float = ALIGNMENT_TOLERANCE,
) -> Optional[Dict[str, Any]]:
    """
    Find the record whose timestamp falls inside the segment interval
    [start - tolerance, end + tolerance], preferring the one nearest the
    segment midpoint.

    Unlike _find_nearest (which only considers the segment start timestamp),
    this matches against the FULL transcript segment interval, so a frame
    shown anywhere while the segment is being spoken is still aligned — e.g.
    a slide change that happens 3 s into a 10 s transcript segment now
    attaches that segment to the new slide instead of missing it.

    Returns:
        Best record dict, or None if no record falls in the interval.
    """
    lo = start - tolerance
    hi = end + tolerance
    midpoint = (start + end) / 2.0

    best: Optional[Dict[str, Any]] = None
    best_dist = float("inf")

    for rec in records:
        rec_ts = rec.get(key, 0.0)
        if lo <= rec_ts <= hi:
            dist = abs(rec_ts - midpoint)
            if dist < best_dist:
                best_dist = dist
                best = rec

    return best


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
    merge_segments: bool | None = None,
) -> List[MultimodalChunk]:
    """
    Execute multimodal fusion: align transcript, VLM, and OCR by timestamp.

    V2 semantic chunk merging: when merge_segments=True (or the
    SEMANTIC_CHUNK_MERGE CloudSettings flag is set), adjacent Whisper
    segments are merged into larger semantic chunks using a deterministic
    rolling-window algorithm with sentence-boundary and silence-gap awareness.

    Reads transcript.json, vlm_output.jsonl, and ocr_output.jsonl from
    the lecture's cloud_runtime directory.

    Args:
        lecture_id: Unique lecture identifier.
        cloud_settings: Injected CloudSettings. The SEMANTIC_CHUNK_MERGE
            field controls merging unless overridden by merge_segments.
        merge_segments: Explicit override for merge behavior. If None,
            falls back to cloud_settings.SEMANTIC_CHUNK_MERGE (default False).

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

    # --- Phase 1: Align every transcript segment (atom) ---
    atoms: List[Dict[str, Any]] = []
    for seg in transcript_segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", seg_start)
        seg_id = seg.get("segment_id", 0)

        # Find the VLM caption whose frame falls inside the segment interval
        # [start, end] (expanded by the alignment tolerance).
        visual_context = ""
        nearest_vlm = _find_nearest_in_interval(
            seg_start, seg_end, vlm_ts_records, key="_timestamp",
        )
        if nearest_vlm is not None:
            visual_context = nearest_vlm.get("caption", "")

        # Find the OCR text whose frame falls inside the segment interval.
        ocr_text = ""
        nearest_ocr = _find_nearest_in_interval(
            seg_start, seg_end, ocr_ts_records, key="_timestamp",
        )
        if nearest_ocr is not None:
            ocr_list = nearest_ocr.get("ocr_text", [])
            if isinstance(ocr_list, list):
                ocr_text = " ".join(ocr_list)
            else:
                ocr_text = str(ocr_list)

        atoms.append({
            "segment_id": seg_id,
            "start": seg_start,
            "end": seg_end,
            "text": seg.get("text", ""),
            "visual_context": visual_context,
            "ocr_text": ocr_text,
        })

    if not atoms:
        raise ValueError("A6: Fusion produced zero atoms.")

    # --- Phase 2: Merge atoms into semantic chunk groups (opt-in) ---
    if merge_segments is None:
        merge_segments = getattr(cloud_settings, "SEMANTIC_CHUNK_MERGE", False) \
            if cloud_settings is not None else False

    if merge_segments:
        groups = _merge_atoms(atoms)
        logger.info(
            "A6: Merged %d atoms into %d semantic chunks",
            len(atoms), len(groups),
        )
    else:
        groups = [[a] for a in atoms]

    # --- Phase 3: Build validated MultimodalChunk instances ---
    chunks: List[MultimodalChunk] = []

    for chunk_num, group in enumerate(groups, start=1):
        first = group[0]
        last = group[-1]

        # Merge transcript text — join with space, preserving natural flow.
        transcript = " ".join(a["text"] for a in group).strip()

        # Visual/OCR from the FIRST atom's aligned frame (the slide active
        # when the chunk begins). Later atoms with a new slide frame are
        # captured by the next chunk.
        visual_context = first["visual_context"]
        ocr_text = first["ocr_text"]

        chunk_id = f"{lecture_id}_chunk_{chunk_num:06d}"

        chunk = MultimodalChunk(
            lecture_id=lecture_id,
            chunk_id=chunk_id,
            timestamp=round(first["start"], 3),
            transcript=transcript,
            visual_context=visual_context,
            ocr_text=ocr_text,
            start_time=round(first["start"], 3),
            end_time=round(last["end"], 3),
            segment_ids=[a["segment_id"] for a in group],
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


# ── Deterministic semantic chunk merge ──────────────────────────────────

def _merge_atoms(
    atoms: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """
    Merge atomic (Whisper-segment) records into semantic chunk groups using
    a deterministic rolling-window algorithm.

    Merge decisions are made on three criteria, evaluated greedily left-to-right:

    1. **Hard cap** (MAX_CHUNK_CHARS): if adding the next atom would exceed
       the maximum, the current group is flushed (unless it is below the
       minimum, in which case the next atom is absorbed to avoid a tiny orphan).

    2. **Strong boundary at content threshold** (MIN_CHUNK_CHARS): a large
       silence gap (>SILENCE_GAP_THRESHOLD) is a topic boundary; a sentence
       ending at or above TARGET_CHUNK_CHARS is a natural segment boundary.
       Both only trigger if the current group has at least MIN_CHUNK_CHARS
       content (otherwise the merge continues).

    3. **Default grow**: absent any boundary signal, the next atom is appended
       to the current group regardless of length.

    This algorithm is fully deterministic — given the same atoms it always
    produces the exact same groups. No ML, no randomness, no external state.

    Each atom dict must have keys:
        segment_id : int
        start      : float  (seconds)
        end        : float  (seconds)
        text       : str

    Args:
        atoms: List of aligned atomic records (one per Whisper segment).

    Returns:
        List of groups, where each group is a list of atom dicts.
    """
    if not atoms:
        return []

    groups: List[List[Dict[str, Any]]] = [[atoms[0]]]

    for atom in atoms[1:]:
        current = groups[-1]
        chars_now = sum(len(a["text"]) for a in current)
        chars_next = len(atom["text"])
        gap = atom["start"] - current[-1]["end"]

        # ── Check last atom's text for sentence-ending punctuation ──
        last_text = current[-1]["text"].strip()
        sentence_boundary = bool(last_text and last_text[-1] in (".", "!", "?"))

        # 1) Hard cap — must break.
        if chars_now + chars_next > MAX_CHUNK_CHARS:
            if chars_now >= MIN_CHUNK_CHARS:
                groups.append([atom])
            else:
                # Current group still too small — absorb to avoid a 40-char orphan.
                current.append(atom)
            continue

        # 2) Strong boundary at content threshold.
        if chars_now >= MIN_CHUNK_CHARS:
            should_break = False

            # Large silence gap = topic boundary.
            if gap > SILENCE_GAP_THRESHOLD:
                should_break = True
            # At or above target + sentence-ending punctuation = natural break.
            elif chars_now >= TARGET_CHUNK_CHARS and sentence_boundary:
                should_break = True

            if should_break:
                groups.append([atom])
                continue

        # 3) Default: keep growing.
        current.append(atom)

    return groups
