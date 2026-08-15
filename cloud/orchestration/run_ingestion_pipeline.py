# cloud/orchestration/run_ingestion_pipeline.py
# Pipeline orchestrator for Stages A1–C2.
#
# Runs the complete cloud ingestion pipeline for a single lecture.
# Full pipeline: A1→A2→A3→A4→A5→A6→A7→A8→A9→Manifest→B0→B1→C1→C2
#
# Optimizations:
#   - A2 (Whisper, GPU) and A3 (Frames, CPU/IO) run concurrently.
#   - Qwen2.5-7B-Instruct loaded once via Transformers, shared across A7/A8/A9/B1.
#   - BGE embeddings computed once in A7 and reused by B0.
#   - In-memory artifact cache eliminates redundant disk reads.
#
# Usage:
#   python -m cloud.orchestration.run_ingestion_pipeline \
#       --lecture-id lec_001 \
#       --video-path /path/to/lecture.mp4

# ----------------------------------------------------------------------------
# TensorFlow / protobuf workaround (mirrors the FRAMEWORK GUARD in run_kaggle.py)
# ----------------------------------------------------------------------------
# Kaggle ships TensorFlow pre-installed. TF's generated protobuf code requires
# a newer protobuf (google.protobuf.runtime_version), but this project pins
# protobuf==3.20.3 for PaddlePaddle 2.6.x compatibility (Paddle aborts at import
# with newer protobuf). transformers probes for TF via importlib.util.find_spec()
# and, when it finds it, imports it while loading Qwen2-VL
# (transformers.image_transforms) — that import crashes with
# "cannot import name 'runtime_version' from 'google.protobuf'".
#
# Marking tensorflow as "imported but disabled" makes find_spec("tensorflow")
# return None, so transformers stays on its PyTorch-only path. This runs at the
# top of this module so it takes effect before any stage imports transformers
# (A4 Qwen2-VL, shared LLM backend, A7/B0 sentence-transformers, ...). This
# subprocess does NOT inherit sys.modules from the parent runner (run_kaggle.py),
# which is why the guard is repeated here.
import sys

sys.modules.setdefault("tensorflow", None)

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import CloudSettings

logger = logging.getLogger(__name__)


def run_pipeline(lecture_id: str, video_path: str) -> dict:
    """
    Execute Stages A1–C2 in sequence for a single lecture.

    Pipeline order:
        A1 Metadata → [A2 Whisper ‖ A3 Frames] → A4 Qwen2-VL → A5 OCR →
        A6 Fusion → A7 Segmentation → A8 Entity Extraction →
        A9 Relation Extraction → Manifest → B0 Embeddings →
        B1 Triplets → C1 Validator → C2 Exporter

    Args:
        lecture_id: Unique lecture identifier.
        video_path: Path to the source video file.

    Returns:
        Summary dict with counts from each stage.
    """
    settings = CloudSettings()
    video = Path(video_path)

    logger.info("=" * 60)
    logger.info("LECTUREMIND V2 — Cloud Ingestion Pipeline (A1–C2)")
    logger.info("Lecture: %s", lecture_id)
    logger.info("Video:   %s", video)
    logger.info("Output:  %s", settings.lecture_dir(lecture_id))
    logger.info("=" * 60)

    # --- A1: Video Metadata ---
    logger.info("Stage A1: Extracting video metadata...")
    from cloud.ingestion.metadata.metadata_extractor import extract_metadata
    metadata = extract_metadata(lecture_id, video, cloud_settings=settings)
    logger.info("A1 complete: duration=%.1fs, fps=%.1f", metadata.duration, metadata.fps)

    # --- A2 + A3: Run concurrently (Whisper=GPU, Frames=CPU/IO) ---
    logger.info("Stage A2+A3: Launching Whisper and Frame extraction concurrently...")
    from cloud.ingestion.whisper_pipeline.transcriber import transcribe
    from cloud.ingestion.frame_extraction.frame_extractor import extract_frames

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a2 = executor.submit(transcribe, lecture_id, video, cloud_settings=settings)
        future_a3 = executor.submit(extract_frames, lecture_id, video, cloud_settings=settings)
        transcript_segments = future_a2.result()
        frames = future_a3.result()

    logger.info("A2 complete: %d transcript segments", len(transcript_segments))
    logger.info("A3 complete: %d key frames", len(frames))

    # --- A4: Visual Understanding (Qwen2-VL) ---
    logger.info("Stage A4: Running Qwen2-VL visual understanding...")
    from cloud.ingestion.qwen_pipeline.visual_understanding import run_visual_understanding
    frames_data = [f.model_dump() for f in frames]
    vlm_count = run_visual_understanding(lecture_id, frames_data, cloud_settings=settings)
    logger.info("A4 complete: %d VLM captions", vlm_count)
    logger.info("[DIAGNOSTIC] Stage A4 completed")

    # --- A5: OCR ---
    logger.info("[DIAGNOSTIC] Starting Stage A5")
    logger.info("Stage A5: Running PaddleOCR...")
    logger.info("[DIAGNOSTIC] Entering run_ocr()")
    from cloud.ingestion.paddleocr_pipeline.ocr_engine import run_ocr
    ocr_count = run_ocr(lecture_id, frames_data, cloud_settings=settings)
    logger.info("A5 complete: %d OCR records", ocr_count)

    # --- A6: Multimodal Fusion ---
    logger.info("Stage A6: Running multimodal fusion (master contract)...")
    from cloud.ingestion.fusion.multimodal_fusion import fuse
    merge_segments = getattr(settings, "SEMANTIC_CHUNK_MERGE", False)
    chunks = fuse(lecture_id, cloud_settings=settings, merge_segments=merge_segments)
    if merge_segments:
        logger.info("A6 complete: %d semantic chunks (V2 merge enabled)", len(chunks))
    else:
        logger.info("A6 complete: %d multimodal chunks (1:1 mode)", len(chunks))

    # --- Load shared LLM backend ONCE for A7/A8/A9/B1 ---
    shared_llm = None
    try:
        from cloud.orchestration.llm_backend import load_shared_llm
        shared_llm = load_shared_llm()
        logger.info("Shared LLM backend (Transformers) loaded successfully.")
    except (ImportError, Exception) as exc:
        logger.warning("Shared LLM backend unavailable (%s), stages will use fallbacks.", exc)

    # Create llm_loader closure for injection.
    def _llm_loader():
        return shared_llm

    # --- A7: Topic Segmentation ---
    logger.info("Stage A7: Running topic segmentation...")
    from cloud.segmentation.segmenter import segment
    if shared_llm is not None:
        segments = segment(lecture_id, cloud_settings=settings,
                           title_generator_loader=_llm_loader)
    else:
        segments = segment(lecture_id, cloud_settings=settings)
    logger.info("A7 complete: %d segments", len(segments))

    # --- A8: Entity Extraction ---
    logger.info("Stage A8: Extracting entities...")
    from cloud.extraction.entity_extractor import extract_entities
    if shared_llm is not None:
        entities = extract_entities(lecture_id, cloud_settings=settings,
                                    llm_loader=_llm_loader)
    else:
        entities = extract_entities(lecture_id, cloud_settings=settings)
    logger.info("A8 complete: %d entities", len(entities))

    # --- A9: Relation Extraction ---
    logger.info("Stage A9: Extracting relations...")
    from cloud.extraction.relation_extractor import extract_relations
    if shared_llm is not None:
        relations = extract_relations(lecture_id, cloud_settings=settings,
                                      llm_loader=_llm_loader)
    else:
        relations = extract_relations(lecture_id, cloud_settings=settings)
    logger.info("A9 complete: %d relations", len(relations))

    # --- Manifest Generation ---
    logger.info("Manifest: Building manifest.json...")
    from cloud.packaging.manifest_builder import build_manifest
    manifest = build_manifest(lecture_id, cloud_settings=settings)
    logger.info("Manifest complete: %d chunks, %d segments, %d entities, %d relations",
                manifest.chunk_count, manifest.segment_count,
                manifest.entity_count, manifest.relation_count)

    # --- B0: Embedding Generation ---
    # Reuse cached embeddings from A7 if available; otherwise compute fresh.
    logger.info("Stage B0: Generating embeddings (bge-large-en-v1.5)...")
    from cloud.embeddings.embedding_generator import generate_embeddings
    lecture_dir = Path(settings.lecture_dir(lecture_id))
    cached_embeddings_path = lecture_dir / "embeddings.npy"
    cached_ids_path = lecture_dir / "embedding_ids.json"
    if cached_embeddings_path.exists() and cached_ids_path.exists():
        # A7 already saved embeddings — skip recomputation.
        import numpy as np
        cached = np.load(str(cached_embeddings_path))
        embedding_count = cached.shape[0]
        logger.info("B0: Reusing %d cached embeddings from A7.", embedding_count)
    else:
        embedding_count = generate_embeddings(lecture_id, cloud_settings=settings)
    logger.info("B0 complete: %d embeddings", embedding_count)

    # --- B1: Triplet Generation ---
    logger.info("Stage B1: Generating reranker training triplets...")
    from cloud.training.triplet_generator import generate_triplets
    if shared_llm is not None:
        triplets = generate_triplets(lecture_id, cloud_settings=settings,
                                     llm_loader=_llm_loader)
    else:
        triplets = generate_triplets(lecture_id, cloud_settings=settings)
    logger.info("B1 complete: %d triplets", len(triplets))

    # --- GPU Memory Cleanup: Release shared LLM after all text gen stages ---
    if shared_llm is not None:
        try:
            shared_llm.cleanup()
        except Exception:
            pass
        shared_llm = None

    # --- B2: Reranker Fine-tuning (optional, default OFF) ---
    # Reuses the existing trainer. Non-fatal: a training failure must never
    # fail the package; the exporter excludes reranker_model/ anyway.
    if getattr(settings, "ENABLE_PIPELINE_RERANKER_TRAINING", False):
        logger.info("Stage B2: Fine-tuning reranker (flag enabled)...")
        try:
            from cloud.training.reranker_trainer import train_reranker
            metrics = train_reranker(lecture_id, cloud_settings=settings)
            logger.info("B2 complete: %s", metrics)
        except Exception as exc:
            logger.warning("B2: Reranker training failed (non-fatal): %s", exc)


    # --- C1: Validation ---
    logger.info("Stage C1: Validating knowledge package...")
    from cloud.packaging.validator import validate_package
    validation_result = validate_package(lecture_id, cloud_settings=settings)
    logger.info("C1 complete: status=%s", validation_result["status"])

    # --- C2: Export ---
    logger.info("Stage C2: Exporting knowledge_package.zip...")
    from cloud.packaging.exporter import export_package
    zip_path = export_package(lecture_id, cloud_settings=settings)
    logger.info("C2 complete: %s", zip_path)

    logger.info("=" * 60)
    logger.info("Pipeline A1–C2 complete. knowledge_package.zip ready.")
    logger.info("=" * 60)

    summary = {
        "lecture_id": lecture_id,
        "duration": metadata.duration,
        "transcript_segments": len(transcript_segments),
        "key_frames": len(frames),
        "vlm_captions": vlm_count,
        "ocr_records": ocr_count,
        "multimodal_chunks": len(chunks),
        "segments": len(segments),
        "entities": len(entities),
        "relations": len(relations),
        "embeddings": embedding_count,
        "triplets": len(triplets),
        "validation_status": validation_result["status"],
        "knowledge_package": str(zip_path),
    }
    return summary


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="LECTUREMIND V2 — Cloud Ingestion A1–C2")
    parser.add_argument("--lecture-id", required=True, help="Unique lecture identifier")
    parser.add_argument("--video-path", required=True, help="Path to lecture video file")
    args = parser.parse_args()

    summary = run_pipeline(args.lecture_id, args.video_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
