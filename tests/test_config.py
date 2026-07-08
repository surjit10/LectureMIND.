"""Tests for config.py — three-class settings split (SharedSettings, CloudSettings, LocalSettings)."""
import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# SharedSettings
# ---------------------------------------------------------------------------

def test_shared_settings_defaults():
    from config import shared_settings
    assert shared_settings.EMBEDDING_DIMENSION == 1024


def test_shared_embedding_dimension_is_1024():
    from config import SharedSettings
    s = SharedSettings(EMBEDDING_DIMENSION=1024)
    assert s.EMBEDDING_DIMENSION == 1024


def test_shared_embedding_dimension_768_rejected():
    """768 is the old bge-base model dimension and must be rejected."""
    from config import SharedSettings
    with pytest.raises(ValidationError, match="1024"):
        SharedSettings(EMBEDDING_DIMENSION=768)


def test_shared_embedding_dimension_arbitrary_rejected():
    from config import SharedSettings
    with pytest.raises(ValidationError, match="1024"):
        SharedSettings(EMBEDDING_DIMENSION=512)


# ---------------------------------------------------------------------------
# CloudSettings
# ---------------------------------------------------------------------------

def test_cloud_settings_defaults():
    from config import cloud_settings
    assert cloud_settings.KAGGLE_OUTPUT_ROOT == "/kaggle/working"
    assert cloud_settings.KAFKA_BROKER == "localhost:9092"
    assert "{lecture_id}" in cloud_settings.LECTURE_OUTPUT_DIR
    assert "{lecture_id}" in cloud_settings.FRAME_DIR
    assert "{lecture_id}" in cloud_settings.LOG_DIR


def test_cloud_lecture_dir_resolution():
    from config import CloudSettings
    cs = CloudSettings()
    resolved = cs.lecture_dir("lec_001")
    assert resolved == "cloud_runtime/lectures/lec_001/"
    assert "{lecture_id}" not in resolved


def test_cloud_frame_dir_resolution():
    from config import CloudSettings
    cs = CloudSettings()
    resolved = cs.frame_dir("lec_001")
    assert resolved == "cloud_runtime/lectures/lec_001/frames/"


def test_cloud_log_dir_resolution():
    from config import CloudSettings
    cs = CloudSettings()
    resolved = cs.log_dir("lec_001")
    assert resolved == "cloud_runtime/lectures/lec_001/logs/"


# ---------------------------------------------------------------------------
# LocalSettings
# ---------------------------------------------------------------------------

def test_local_settings_defaults():
    from config import local_settings
    assert local_settings.NEO4J_URI == "bolt://localhost:7687"
    assert local_settings.QDRANT_URL == "http://localhost:6333"
    assert local_settings.OLLAMA_MODEL == "qwen2.5:3b"
    assert local_settings.LOCAL_MODEL_DIR == "local_runtime/models/"
    assert local_settings.TRANSFER_DIR == "transfer/"
    assert local_settings.LOCAL_LOG_DIR == "local_runtime/logs/"
    assert local_settings.LOCAL_CACHE_DIR == "local_runtime/cache/"
    assert local_settings.LOCAL_UPLOAD_DIR == "local_runtime/uploads/"


# ---------------------------------------------------------------------------
# Isolation checks
# ---------------------------------------------------------------------------

def test_cloud_settings_has_no_neo4j():
    """Cloud settings must not contain local database config."""
    from config import CloudSettings
    cs = CloudSettings()
    assert not hasattr(cs, "NEO4J_URI")
    assert not hasattr(cs, "QDRANT_URL")
    assert not hasattr(cs, "OLLAMA_MODEL")


def test_local_settings_has_no_kaggle():
    """Local settings must not contain cloud config."""
    from config import LocalSettings
    ls = LocalSettings()
    assert not hasattr(ls, "KAGGLE_OUTPUT_ROOT")
    assert not hasattr(ls, "KAFKA_BROKER")
    assert not hasattr(ls, "LECTURE_OUTPUT_DIR")


# ---------------------------------------------------------------------------
# PipelineStatus enum (unchanged, kept for coverage)
# ---------------------------------------------------------------------------

def test_pipeline_status_enum():
    from schemas.enums import PipelineStatus
    expected = {
        "UPLOADED", "PROCESSING", "SEGMENTING", "EXTRACTING_ENTITIES",
        "BUILDING_GRAPH", "GENERATING_EMBEDDINGS", "TRAINING_RERANKER",
        "PACKAGING", "READY", "FAILED",
    }
    actual = {s.value for s in PipelineStatus}
    assert actual == expected
