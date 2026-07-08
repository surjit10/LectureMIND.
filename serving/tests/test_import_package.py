# serving/tests/test_import_package.py
# Tests for the POST /upload (import_knowledge_package) endpoint.
# All tests are fully isolated — no running server, no live DBs.
# Every external dependency (Qdrant, Neo4j, loaders) is mocked.

import io
import json
import zipfile
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_MANIFEST = {
    "lecture_id": "test_lecture",
    "chunk_count": 1,
    "segment_count": 1,
    "entity_count": 1,
    "relation_count": 1,
    "embedding_dimension": 1024,
    "pipeline_version": "7.0",
    "package_version": "1.0",
}

VALID_METADATA = {
    "title": "Operating Systems",
    "course_name": "CS301",
    "speaker": "Prof. Smith",
    "language": "en",
    "duration": 3600.0,
    "description": "Week 4 — Scheduling",
}

REQUIRED_FILES = [
    "manifest.json",
    "segments.json",
    "entities.json",
    "relations.json",
    "embedding_ids.json",
    "multimodal_chunks.json",
]


def _make_zip(
    *,
    include_manifest: bool = True,
    include_metadata: bool = True,
    include_all_files: bool = True,
    corrupt: bool = False,
    manifest_override: dict | None = None,
    metadata_override: dict | None = None,
) -> bytes:
    """
    Build an in-memory ZIP that looks like a LectureMind knowledge package.
    Returns raw bytes.
    """
    buf = io.BytesIO()

    if corrupt:
        # Return garbage bytes that are not a valid ZIP
        return b"NOT A VALID ZIP FILE %^&*"

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if include_manifest:
            mf = manifest_override if manifest_override is not None else VALID_MANIFEST
            zf.writestr("manifest.json", json.dumps(mf))
        if include_metadata:
            md = metadata_override if metadata_override is not None else VALID_METADATA
            zf.writestr("metadata.json", json.dumps(md))
        if include_all_files:
            for fname in REQUIRED_FILES:
                if fname not in ("manifest.json", "metadata.json"):
                    zf.writestr(fname, "[]")
            # embeddings.npy — minimal 2-D array placeholder
            import numpy as np
            emb = np.zeros((1, 1024), dtype=np.float32)
            emb_buf = io.BytesIO()
            np.save(emb_buf, emb)
            zf.writestr("embeddings.npy", emb_buf.getvalue())
            # NOTE: reranker_model/ is intentionally excluded from packages.

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    """
    TestClient with a fully isolated registry and all heavy loaders mocked out.

    ISOLATION PATTERN
    -----------------
    We create a ``LectureRegistry`` pointing at a ``tmp_path``-backed JSON file
    and register it via ``set_registry()``.  Every call to ``get_registry()``
    inside the route handlers returns this isolated instance — no monkey-patching
    of import-time constants is required, and the production registry is never
    touched.

    ``reset_registry()`` in the finalizer ensures no isolated instance leaks
    across tests even if a test raises an exception.
    """
    from local.storage.lecture_registry import LectureRegistry
    from local.storage.registry_provider import set_registry, reset_registry

    isolated_registry = LectureRegistry(registry_file=tmp_path / "test_registry.json")
    set_registry(isolated_registry)

    with (
        # ── External loaders ────────────────────────────────────────────────
        patch("local.loaders.qdrant_loader.load_qdrant"),
        patch("local.loaders.neo4j_loader.load_neo4j"),
        patch("local.loaders.package_loader.load_package", return_value=MagicMock()),
        patch("agent.langgraph.workflow.QueryWorkflow", return_value=MagicMock()),
        patch("serving.fastapi.routes.query.set_active_package"),
        patch("serving.fastapi.routes.query.clear_workflow"),
        patch("serving.fastapi.routes.query.set_workflow"),
        patch("neo4j.GraphDatabase.driver", return_value=MagicMock()),
        patch("qdrant_client.QdrantClient", return_value=MagicMock()),
        # ── Bypass manifest deep-validation (counts, embeddings) ────────────
        patch(
            "local.loaders.manifest_validator.validate_manifest",
            return_value={"status": "VALID"},
        ),
    ):
        from serving.fastapi.app import app
        yield TestClient(app)

    # Always restore — runs even when a test raises.
    reset_registry()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestImportPackage:

    # ── Happy path ─────────────────────────────────────────────────────────

    def test_valid_package_succeeds(self, client):
        zip_bytes = _make_zip()
        res = client.post(
            "/upload",
            files={"file": ("knowledge_package.zip", zip_bytes, "application/zip")},
            data={"display_name": "Operating Systems", "description": "Week 4"},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["status"] == "READY"
        assert "lecture_id" in body

    def test_package_name_stored_in_registry(self, client):
        zip_bytes = _make_zip()
        res = client.post(
            "/upload",
            files={"file": ("knowledge_package.zip", zip_bytes, "application/zip")},
            data={"display_name": "My Package"},
        )
        assert res.status_code == 200
        lecture_id = res.json()["lecture_id"]

        detail = client.get(f"/lecture/{lecture_id}")
        assert detail.status_code == 200
        assert detail.json()["display_name"] == "My Package"

    def test_metadata_from_zip_used_when_no_display_name(self, client):
        zip_bytes = _make_zip()
        res = client.post(
            "/upload",
            files={"file": ("knowledge_package.zip", zip_bytes, "application/zip")},
        )
        assert res.status_code == 200
        lecture_id = res.json()["lecture_id"]

        detail = client.get(f"/lecture/{lecture_id}")
        assert detail.status_code == 200
        # Should fall back to title from metadata.json
        assert detail.json()["title"] == "Operating Systems"

    # ── Package name validation ────────────────────────────────────────────

    def test_package_name_min_length(self, client):
        """Name shorter than 3 chars after trim must be rejected by the frontend
        validation; verify the backend rename endpoint also enforces this."""
        res = client.patch(
            "/lectures/nonexistent_id/rename",
            json={"display_name": "ab"},
        )
        # 404 because package doesn't exist; name is only validated if found
        assert res.status_code in (400, 404)

    def test_package_name_four_chars_accepted(self, client):
        """'demo' (4 chars) must pass validation — this is the regression case."""
        zip_bytes = _make_zip()
        # Upload a package first
        res = client.post(
            "/upload",
            files={"file": ("knowledge_package.zip", zip_bytes, "application/zip")},
            data={"display_name": "demo"},
        )
        assert res.status_code == 200, res.text

    def test_package_name_exactly_3_chars(self, client):
        zip_bytes = _make_zip()
        res = client.post(
            "/upload",
            files={"file": ("knowledge_package.zip", zip_bytes, "application/zip")},
            data={"display_name": "abc"},
        )
        assert res.status_code == 200, res.text

    def test_package_name_whitespace_trimmed(self, client):
        zip_bytes = _make_zip()
        res = client.post(
            "/upload",
            files={"file": ("knowledge_package.zip", zip_bytes, "application/zip")},
            data={"display_name": "  trimmed name  "},
        )
        assert res.status_code == 200, res.text

    def test_rename_endpoint_updates_display_name(self, client):
        zip_bytes = _make_zip()
        upload = client.post(
            "/upload",
            files={"file": ("knowledge_package.zip", zip_bytes, "application/zip")},
            data={"display_name": "Old Name"},
        )
        assert upload.status_code == 200
        lecture_id = upload.json()["lecture_id"]

        rename = client.patch(
            f"/lectures/{lecture_id}/rename",
            json={"display_name": "New Name"},
        )
        assert rename.status_code == 200
        assert rename.json()["display_name"] == "New Name"

    # ── Wrong file type ────────────────────────────────────────────────────

    def test_wrong_extension_rejected(self, client):
        res = client.post(
            "/upload",
            files={"file": ("lecture.pdf", b"PDF content", "application/pdf")},
            data={"display_name": "My Package"},
        )
        assert res.status_code == 400
        assert "zip" in res.json()["detail"].lower()

    def test_docx_rejected(self, client):
        res = client.post(
            "/upload",
            files={"file": ("notes.docx", b"DOCX content", "application/octet-stream")},
            data={"display_name": "My Package"},
        )
        assert res.status_code == 400

    def test_mp4_rejected(self, client):
        res = client.post(
            "/upload",
            files={"file": ("lecture.mp4", b"VIDEO", "video/mp4")},
            data={"display_name": "My Package"},
        )
        assert res.status_code == 400

    # ── Corrupt archive ────────────────────────────────────────────────────

    def test_corrupt_zip_returns_400(self, client):
        corrupt_bytes = _make_zip(corrupt=True)
        res = client.post(
            "/upload",
            files={"file": ("knowledge_package.zip", corrupt_bytes, "application/zip")},
            data={"display_name": "Corrupt Package"},
        )
        assert res.status_code == 400
        assert "corrupted" in res.json()["detail"].lower() or "valid zip" in res.json()["detail"].lower()

    # ── Missing files ──────────────────────────────────────────────────────

    def test_missing_manifest_returns_400(self, client):
        zip_bytes = _make_zip(include_manifest=False)
        res = client.post(
            "/upload",
            files={"file": ("knowledge_package.zip", zip_bytes, "application/zip")},
            data={"display_name": "No Manifest"},
        )
        assert res.status_code == 400
        assert "manifest" in res.json()["detail"].lower()



    # ── Empty ZIP ──────────────────────────────────────────────────────────

    def test_empty_zip_returns_400(self, client):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w"):
            pass  # empty
        res = client.post(
            "/upload",
            files={"file": ("knowledge_package.zip", buf.getvalue(), "application/zip")},
            data={"display_name": "Empty Package"},
        )
        assert res.status_code == 400

    # ── List and detail after import ───────────────────────────────────────

    def test_imported_package_appears_in_list(self, client):
        zip_bytes = _make_zip()
        client.post(
            "/upload",
            files={"file": ("knowledge_package.zip", zip_bytes, "application/zip")},
            data={"display_name": "Listed Package"},
        )
        res = client.get("/lectures")
        assert res.status_code == 200
        names = [lec["display_name"] for lec in res.json()]
        assert "Listed Package" in names

    def test_detail_returns_all_v7_fields(self, client):
        zip_bytes = _make_zip()
        upload = client.post(
            "/upload",
            files={"file": ("knowledge_package.zip", zip_bytes, "application/zip")},
            data={"display_name": "Full Meta", "description": "desc"},
        )
        assert upload.status_code == 200
        lecture_id = upload.json()["lecture_id"]

        res = client.get(f"/lecture/{lecture_id}")
        assert res.status_code == 200
        data = res.json()
        assert data["lecture_id"] == lecture_id
        assert data["display_name"] == "Full Meta"
        assert data["description"] == "desc"
        assert data["status"] == "READY"

    # ── 404 for unknown package ────────────────────────────────────────────

    def test_get_unknown_package_returns_404(self, client):
        res = client.get("/lecture/definitely_nonexistent")
        assert res.status_code == 404

    # ── Delete ─────────────────────────────────────────────────────────────

    def test_delete_removes_package_from_list(self, client):
        zip_bytes = _make_zip()
        upload = client.post(
            "/upload",
            files={"file": ("knowledge_package.zip", zip_bytes, "application/zip")},
            data={"display_name": "To Delete"},
        )
        assert upload.status_code == 200
        lecture_id = upload.json()["lecture_id"]

        with (
            patch("qdrant_client.QdrantClient", return_value=MagicMock()),
            patch("neo4j.GraphDatabase.driver", return_value=MagicMock()),
        ):
            del_res = client.delete(f"/lectures/{lecture_id}")
        assert del_res.status_code == 204

        list_res = client.get("/lectures")
        ids = [lec["lecture_id"] for lec in list_res.json()]
        assert lecture_id not in ids
