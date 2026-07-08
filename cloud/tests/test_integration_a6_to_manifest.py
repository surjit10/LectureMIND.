# cloud/tests/test_integration_a6_to_manifest.py
# Integration test: A6 → A7 → A8 → A9 → Manifest → B1.
#
# Verifies the full data flow from multimodal_chunks.json through
# segmentation, entity extraction, relation extraction, manifest
# generation, and triplet generation.
#
# All ML models are mocked — no downloads.

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from config import CloudSettings, SharedSettings
from schemas.chunk import MultimodalChunk
from schemas.segment import LectureSegment
from schemas.entity import Entity
from schemas.relation import Relation
from schemas.manifest import Manifest
from schemas.triplet import RerankerTriplet


@pytest.fixture
def cloud_settings(tmp_path):
    return CloudSettings(
        LECTURE_OUTPUT_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/"),
        FRAME_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/frames/"),
        LOG_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/logs/"),
    )


@pytest.fixture
def setup_a6_output(cloud_settings):
    """Create A6 output: multimodal_chunks.json with realistic content."""
    lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
    lecture_dir.mkdir(parents=True, exist_ok=True)

    chunks = [
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000001",
         "timestamp": 10.0, "transcript": "BFS uses queue for breadth first traversal",
         "visual_context": "BFS traversal diagram showing queue operations",
         "ocr_text": "Breadth First Search O(V+E)"},
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000002",
         "timestamp": 20.0, "transcript": "BFS explores nodes level by level",
         "visual_context": "BFS level order diagram", "ocr_text": "Queue FIFO"},
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000003",
         "timestamp": 30.0, "transcript": "BFS shortest path in unweighted graphs",
         "visual_context": "BFS shortest path tree", "ocr_text": "Shortest Path Algorithm"},
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000004",
         "timestamp": 120.0, "transcript": "DFS uses stack for depth first traversal",
         "visual_context": "DFS tree structure with backtracking",
         "ocr_text": "Depth First Search O(V+E)"},
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000005",
         "timestamp": 130.0, "transcript": "DFS explores as far as possible before backtracking",
         "visual_context": "DFS recursion call stack", "ocr_text": "Stack LIFO"},
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000006",
         "timestamp": 140.0, "transcript": "DFS can detect cycles in directed graphs",
         "visual_context": "DFS cycle detection algorithm",
         "ocr_text": "Cycle Detection Back Edge"},
    ]
    (lecture_dir / "multimodal_chunks.json").write_text(json.dumps(chunks, indent=2))
    return lecture_dir


# --- Mock factories ---

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
        titles = ["BFS Graph Traversal", "DFS and Backtracking"]
        return [titles[i % len(titles)] for i in range(len(prompts))]
    mock.generate.side_effect = gen
    return mock


def _mock_entity_llm():
    mock = MagicMock()
    entity_json = json.dumps([
        {"name": "Breadth First Search", "type": "Algorithm"},
        {"name": "Depth First Search", "type": "Algorithm"},
        {"name": "O(V+E)", "type": "Formula"},
        {"name": "Queue", "type": "Concept"},
        {"name": "Stack", "type": "Concept"},
        {"name": "BFS traversal diagram", "type": "Diagram"},
    ])
    def gen(prompts, **kwargs):
        return [entity_json for _ in prompts]
    mock.generate.side_effect = gen
    return mock


def _mock_relation_llm(entity_ids):
    mock = MagicMock()
    # Build relations using actual entity_ids.
    if len(entity_ids) >= 3:
        relation_json = json.dumps([
            {"source_entity_id": entity_ids[0], "relation": "PREREQUISITE_OF",
             "target_entity_id": entity_ids[1]},
            {"source_entity_id": entity_ids[2], "relation": "DERIVED_FROM",
             "target_entity_id": entity_ids[0]},
        ])
    else:
        relation_json = json.dumps([
            {"source_entity_id": entity_ids[0], "relation": "EXPLAINS",
             "target_entity_id": entity_ids[-1]},
        ])
    def gen(prompts, **kwargs):
        return [relation_json for _ in prompts]
    mock.generate.side_effect = gen
    return mock


def _mock_query_llm():
    mock = MagicMock()
    queries_json = json.dumps([
        "What is BFS?",
        "When should BFS be preferred over DFS?",
        "Why does BFS use a queue?",
        "What is the time complexity of BFS?",
    ])
    def gen(prompts, **kwargs):
        return [queries_json for _ in prompts]
    mock.generate.side_effect = gen
    return mock


class TestIntegrationA6ToManifest:
    """
    Integration test verifying the full pipeline:
    A6 output → A7 Segmentation → A8 Entity Extraction →
    A9 Relation Extraction → Manifest → B1 Triplets.
    """

    def test_full_pipeline_flow(self, cloud_settings, setup_a6_output):
        """Execute A7 → A8 → A9 → Manifest → B1 in sequence and validate all outputs."""
        lecture_dir = setup_a6_output

        # --- A7: Segmentation ---
        from cloud.segmentation.segmenter import segment

        segments = segment(
            "lec_001",
            cloud_settings=cloud_settings,
            min_chunks=2,
            embedding_model_loader=lambda: _mock_embedding_model(),
            title_generator_loader=lambda: _mock_title_generator(),
        )

        assert len(segments) >= 1
        assert all(isinstance(s, LectureSegment) for s in segments)
        assert (lecture_dir / "segments.json").exists()
        assert (lecture_dir / "chunk_segment_map.json").exists()

        # All 6 chunks must be assigned.
        total_chunks = sum(len(s.chunks) for s in segments)
        assert total_chunks == 6

        # --- A8: Entity Extraction ---
        from cloud.extraction.entity_extractor import extract_entities

        entities = extract_entities(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _mock_entity_llm(),
        )

        assert len(entities) >= 1
        assert all(isinstance(e, Entity) for e in entities)
        assert (lecture_dir / "entities.json").exists()

        # All entity_ids must be unique.
        entity_ids = [e.entity_id for e in entities]
        assert len(entity_ids) == len(set(entity_ids))

        # --- A9: Relation Extraction ---
        from cloud.extraction.relation_extractor import extract_relations

        relations = extract_relations(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _mock_relation_llm(entity_ids),
        )

        assert len(relations) >= 1
        assert all(isinstance(r, Relation) for r in relations)
        assert (lecture_dir / "relations.json").exists()

        # All relation entity_ids must reference actual entities.
        entity_id_set = set(entity_ids)
        for rel in relations:
            assert rel.source_entity_id in entity_id_set
            assert rel.target_entity_id in entity_id_set
            assert rel.source_entity_id != rel.target_entity_id

        # --- Manifest Generation ---
        from cloud.packaging.manifest_builder import build_manifest

        manifest = build_manifest("lec_001", cloud_settings=cloud_settings)

        assert isinstance(manifest, Manifest)
        assert manifest.lecture_id == "lec_001"
        assert manifest.chunk_count == 6
        assert manifest.segment_count == len(segments)
        assert manifest.entity_count == len(entities)
        assert manifest.relation_count == len(relations)
        assert manifest.embedding_dimension == 1024
        assert manifest.embedding_model == cloud_settings.BGE_MODEL_PATH
        assert manifest.reranker_version == "BAAI/bge-reranker-base"
        assert (lecture_dir / "manifest.json").exists()

        # --- B1: Triplet Generation ---
        from cloud.training.triplet_generator import generate_triplets

        triplets = generate_triplets(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _mock_query_llm(),
        )

        assert len(triplets) >= 1
        assert all(isinstance(t, RerankerTriplet) for t in triplets)
        assert (lecture_dir / "triplets.json").exists()

        # All triplet fields must be non-empty.
        for t in triplets:
            assert t.query.strip()
            assert t.positive.strip()
            assert t.negative.strip()

    def test_schema_compliance_all_outputs(self, cloud_settings, setup_a6_output):
        """All output files must pass Pydantic schema validation."""
        lecture_dir = setup_a6_output

        # Run stages.
        from cloud.segmentation.segmenter import segment
        from cloud.extraction.entity_extractor import extract_entities
        from cloud.extraction.relation_extractor import extract_relations
        from cloud.packaging.manifest_builder import build_manifest

        segment(
            "lec_001", cloud_settings=cloud_settings, min_chunks=2,
            embedding_model_loader=lambda: _mock_embedding_model(),
            title_generator_loader=lambda: _mock_title_generator(),
        )

        entities = extract_entities(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _mock_entity_llm(),
        )
        entity_ids = [e.entity_id for e in entities]

        extract_relations(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _mock_relation_llm(entity_ids),
        )

        build_manifest("lec_001", cloud_settings=cloud_settings)

        # Validate every output file against its schema.
        chunks_data = json.loads((lecture_dir / "multimodal_chunks.json").read_text())
        for c in chunks_data:
            MultimodalChunk(**c)

        segments_data = json.loads((lecture_dir / "segments.json").read_text())
        for s in segments_data:
            LectureSegment(**s)

        entities_data = json.loads((lecture_dir / "entities.json").read_text())
        for e in entities_data:
            Entity(**e)

        relations_data = json.loads((lecture_dir / "relations.json").read_text())
        for r in relations_data:
            Relation(**r)

        manifest_data = json.loads((lecture_dir / "manifest.json").read_text())
        Manifest(**manifest_data)

    def test_no_data_loss_through_pipeline(self, cloud_settings, setup_a6_output):
        """Original multimodal_chunks.json must not be modified by downstream stages."""
        lecture_dir = setup_a6_output

        # Read original chunks.
        original_chunks = json.loads((lecture_dir / "multimodal_chunks.json").read_text())
        original_count = len(original_chunks)

        # Run full pipeline.
        from cloud.segmentation.segmenter import segment
        from cloud.extraction.entity_extractor import extract_entities
        from cloud.extraction.relation_extractor import extract_relations

        segment(
            "lec_001", cloud_settings=cloud_settings, min_chunks=2,
            embedding_model_loader=lambda: _mock_embedding_model(),
            title_generator_loader=lambda: _mock_title_generator(),
        )
        entities = extract_entities(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _mock_entity_llm(),
        )
        extract_relations(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _mock_relation_llm([e.entity_id for e in entities]),
        )

        # Verify chunks unchanged.
        after_chunks = json.loads((lecture_dir / "multimodal_chunks.json").read_text())
        assert len(after_chunks) == original_count
        assert after_chunks == original_chunks
