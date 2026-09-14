# cloud/tests/test_integration_full_pipeline.py
# FULL PIPELINE DRY-RUN — pre-flight verification for a Kaggle run.
#
# Exercises the EXACT code path the Kaggle orchestrator runs
# (cloud/orchestration/run_ingestion_pipeline.py) with every ML model mocked:
#
#   A6 fuse(merge_segments=True)  — REAL code, semantic chunk merge + provenance
#   A7 segment()                  — REAL code, mocked embedding + title LLM
#   A8 extract_entities()         — REAL code, mocked LLM returns W\_q broken JSON
#   A9 extract_relations()        — REAL code, mocked LLM returns MODIFIES
#   Manifest → B0 cache → B1      — REAL code
#   C1 validate_package()         — REAL code, hard validation on merged chunks
#   C2 export_package()           — REAL code, ZIP with all package files
#
# Purpose: if this passes locally in seconds, the pipeline WIRING is correct
# and the only remaining Kaggle risks are GPU/model-path specific (checked
# separately by scripts/validate_model_paths.py). Run before every Kaggle run:
#
#   .venv/bin/python -m pytest cloud/tests/test_integration_full_pipeline.py -v

import json
import zipfile
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from config import CloudSettings
from schemas.chunk import MultimodalChunk


@pytest.fixture
def cloud_settings(tmp_path):
    return CloudSettings(
        LECTURE_OUTPUT_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/"),
        FRAME_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/frames/"),
        LOG_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/logs/"),
    )


@pytest.fixture
def setup_raw_inputs(cloud_settings):
    """Synthetic A1–A5 outputs: transcript, frames, VLM captions, OCR."""
    lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
    lecture_dir.mkdir(parents=True, exist_ok=True)

    # Whisper segments like a real lecture (median ~49 chars). Long enough
    # that the semantic merge yields 4+ chunks → ≥2 A7 segments → B1 gets
    # sibling negatives.
    segments = [
        {"segment_id": 1, "start": 0.0, "end": 1.2, "text": "Okay, so today we are going to talk about graph traversal algorithms. "},
        {"segment_id": 2, "start": 1.3, "end": 3.0, "text": "We will start with breadth first search, which we call BFS for short. "},
        {"segment_id": 3, "start": 3.2, "end": 5.0, "text": "BFS uses a queue data structure to explore nodes level by level. "},
        {"segment_id": 4, "start": 5.2, "end": 7.0, "text": "The time complexity of BFS is O of V plus E where V is vertices. "},
        {"segment_id": 5, "start": 7.5, "end": 9.0, "text": "Yeah, that is the key point about BFS and why it works so well. "},
        # 2.2 s silence gap → topic boundary between BFS and DFS.
        {"segment_id": 6, "start": 11.2, "end": 13.0, "text": "Next we move on to depth first search, which we call DFS for short, and it uses a stack instead of a queue. "},
        {"segment_id": 7, "start": 13.2, "end": 15.0, "text": "DFS explores as far as possible down one single branch of the tree before it backtracks and tries another path. "},
        {"segment_id": 8, "start": 15.2, "end": 17.0, "text": "DFS is particularly useful for cycle detection in directed graphs and for topological sorting of dependencies. "},
        {"segment_id": 9, "start": 17.2, "end": 19.0, "text": "The space complexity of DFS depends directly on the maximum depth of the recursion stack that gets built. "},
        # 2.5 s silence gap → topic boundary before the worked example.
        {"segment_id": 10, "start": 21.5, "end": 23.0, "text": "Let me show you an example on the board with a small graph right now. "},
        {"segment_id": 11, "start": 23.2, "end": 25.0, "text": "Any questions about the difference between BFS and DFS so far? "},
        {"segment_id": 12, "start": 25.5, "end": 27.0, "text": "Great, we will continue with more graph algorithms in the next part. "},
    ]
    (lecture_dir / "transcript.json").write_text(json.dumps(segments))

    frames = [
        {"frame_id": 1, "timestamp": 0.5, "image_path": "frames/frame_000001.jpg"},
        {"frame_id": 2, "timestamp": 2.0, "image_path": "frames/frame_000002.jpg"},
        {"frame_id": 3, "timestamp": 4.0, "image_path": "frames/frame_000003.jpg"},
        {"frame_id": 4, "timestamp": 9.5, "image_path": "frames/frame_000004.jpg"},
        {"frame_id": 5, "timestamp": 12.0, "image_path": "frames/frame_000005.jpg"},
    ]
    (lecture_dir / "frames.json").write_text(json.dumps(frames))

    vlm_lines = "\n".join(
        json.dumps({"frame_id": f, "caption": cap, "objects": []})
        for f, cap in [
            (1, "Lecture introduction slide"),
            (2, "BFS traversal diagram with queue"),
            (3, "Complexity formula O(V+E)"),
            (4, "DFS tree with stack"),
            (5, "Questions slide"),
        ]
    ) + "\n"
    (lecture_dir / "vlm_output.jsonl").write_text(vlm_lines)

    ocr_lines = "\n".join(
        json.dumps({"frame_id": f, "ocr_text": lines})
        for f, lines in [
            (1, ["Lecture 1", "Graph Algorithms"]),
            (2, ["Breadth First Search", "Queue"]),
            (3, ["O(V+E)"]),
            (4, ["Depth First Search", "Stack"]),
            (5, ["Questions?"]),
        ]
    ) + "\n"
    (lecture_dir / "ocr_output.jsonl").write_text(ocr_lines)

    # A1 metadata (schema: lecture_id, duration, fps — extra=forbid).
    (lecture_dir / "metadata.json").write_text(json.dumps({
        "lecture_id": "lec_001", "duration": 27.0, "fps": 30.0,
    }))

    return lecture_dir


# --- Mock model factories (identical shape to the real backends) ---

def _mock_embedding_model(dim=1024):
    mock = MagicMock()
    def encode_fn(texts, **kwargs):
        n = len(texts)
        embeddings = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            np.random.seed(42 + i)
            embeddings[i] = np.random.randn(dim).astype(np.float32)
            embeddings[i] /= np.linalg.norm(embeddings[i])
        return embeddings
    mock.encode.side_effect = encode_fn
    return mock


def _mock_title_generator():
    mock = MagicMock()
    def gen(prompts, **kwargs):
        return [f"Segment {i+1} topic" for i in range(len(prompts))]
    mock.generate.side_effect = gen
    return mock


def _mock_entity_llm_with_repair():
    """Returns the exact broken JSON the real Qwen model produced on Kaggle:
    'W\\_q' is an invalid escape that used to discard the WHOLE response.
    """
    mock = MagicMock()
    broken_json = json.dumps([
        {"name": "W\\_q", "type": "Variable"},          # invalid escape + bad type
        {"name": "Breadth First Search", "type": "Algorithm"},
        {"name": "Queue", "type": "Concept"},
    ])
    def gen(prompts, **kwargs):
        return [broken_json for _ in prompts]
    mock.generate.side_effect = gen
    return mock


def _mock_relation_llm(entity_ids):
    mock = MagicMock()
    relation_json = json.dumps([
        {"source_entity_id": entity_ids[1], "relation": "MODIFIES",  # invalid type
         "target_entity_id": entity_ids[2]},
        {"source_entity_id": entity_ids[1], "relation": "PREREQUISITE_OF",
         "target_entity_id": entity_ids[0],
         # Evidence rule (2026-09-14): quote is verbatim from the transcript
         # fixture so the evidence-verification gate accepts it.
         "evidence": "BFS uses a queue data structure to explore nodes level by level."},
    ])
    def gen(prompts, **kwargs):
        return [relation_json for _ in prompts]
    mock.generate.side_effect = gen
    return mock


def _mock_query_llm():
    mock = MagicMock()
    # Fenced JSON block — the second real Kaggle failure mode.
    fenced = "```json\n" + json.dumps([
        "What is BFS?", "Why does BFS use a queue?",
    ]) + "\n```"
    def gen(prompts, **kwargs):
        return [fenced for _ in prompts]
    mock.generate.side_effect = gen
    return mock


class TestFullPipelineDryRun:

    def test_a6_to_c2_with_merge_and_repair(self, cloud_settings, setup_raw_inputs):
        """The entire Kaggle pipeline runs end-to-end with the new code paths."""
        lecture_dir = setup_raw_inputs

        # ── A6: Fusion with semantic merge (the Kaggle flag path) ──
        from cloud.ingestion.fusion.multimodal_fusion import fuse
        chunks = fuse("lec_001", cloud_settings=cloud_settings, merge_segments=True)

        # 12 whisper segments → fewer merged chunks (semantic merge).
        assert len(chunks) < 12
        assert len(chunks) >= 3
        assert all(isinstance(c, MultimodalChunk) for c in chunks)
        for c in chunks:
            assert c.start_time <= c.end_time
            assert len(c.segment_ids) >= 1
            assert c.segment_ids == sorted(c.segment_ids)

        # ── A7: Segmentation (mocked embedding + titles) ──
        from cloud.segmentation.segmenter import segment
        segments = segment(
            "lec_001", cloud_settings=cloud_settings, min_chunks=2,
            embedding_model_loader=lambda: _mock_embedding_model(),
            title_generator_loader=lambda: _mock_title_generator(),
        )
        assert len(segments) >= 1
        # A7 also cached embeddings.npy + embedding_ids.json for B0 reuse.
        assert (lecture_dir / "embeddings.npy").exists()
        assert (lecture_dir / "embedding_ids.json").exists()

        # ── A8: Entity extraction with W\_q repair ──
        from cloud.extraction.entity_extractor import extract_entities
        entities = extract_entities(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _mock_entity_llm_with_repair(),
        )
        names = {e.name for e in entities}
        # The W\_q response survived repair; "Variable" was normalized.
        assert "Breadth First Search" in names
        assert (lecture_dir / "entities.json").exists()

        # ── A9: Relation extraction (MODIFIES rejected, valid one kept) ──
        from cloud.extraction.relation_extractor import extract_relations
        relations = extract_relations(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _mock_relation_llm([e.entity_id for e in entities]),
        )
        assert len(relations) >= 1
        assert (lecture_dir / "relations.json").exists()

        # ── Manifest ──
        from cloud.packaging.manifest_builder import build_manifest
        manifest = build_manifest("lec_001", cloud_settings=cloud_settings)
        assert manifest.chunk_count == len(chunks)
        assert (lecture_dir / "manifest.json").exists()

        # ── B1: Triplets with fenced-JSON repair ──
        from cloud.training.triplet_generator import generate_triplets
        triplets = generate_triplets(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _mock_query_llm(),
        )
        assert len(triplets) >= 1
        assert (lecture_dir / "triplets.json").exists()

        # ── C1: Hard validation on the merged-chunk package ──
        from cloud.packaging.validator import validate_package
        summary = validate_package("lec_001", cloud_settings=cloud_settings)
        assert summary["status"] == "VALID"
        assert summary["chunk_count"] == len(chunks)

        # ── C2: Export the ZIP and verify contents ──
        from cloud.packaging.exporter import export_package, PACKAGE_FILES
        zip_path = export_package(
            "lec_001", cloud_settings=cloud_settings,
            transfer_dir=str(lecture_dir.parent / "transfer"),
            skip_validation=True,
        )
        assert zip_path.exists()
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
            for required in PACKAGE_FILES:
                assert required in names, f"Missing from package: {required}"

    def test_a6_without_merge_still_1to1(self, cloud_settings, setup_raw_inputs):
        """Default flag OFF must keep the legacy 1:1 behavior (no regression)."""
        from cloud.ingestion.fusion.multimodal_fusion import fuse
        chunks = fuse("lec_001", cloud_settings=cloud_settings, merge_segments=False)
        assert len(chunks) == 12  # one per whisper segment (legacy 1:1)
        for c in chunks:
            assert len(c.segment_ids) == 1