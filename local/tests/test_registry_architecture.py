# local/tests/test_registry_architecture.py
#
# Tests for the registry architecture hardening:
#   - LectureRegistry isolation guarantee
#   - set_registry / get_registry / reset_registry provider contract
#   - Consistency audit (missing dirs, orphaned dirs, duplicate names)
#   - CRUD lifecycle
#   - Registry file recovery from corrupt JSON
#   - Active-lecture persistence
#   - Production registry is never touched by any test in this file

import json
import pytest
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reg(tmp_path):
    """A fresh isolated LectureRegistry backed by a tmp_path file."""
    from local.storage.lecture_registry import LectureRegistry
    return LectureRegistry(registry_file=tmp_path / "registry.json")


@pytest.fixture(autouse=True)
def isolate_registry(reg):
    """
    Automatically inject the isolated registry before every test and restore
    the default after.  This guarantees get_registry() never returns the
    production singleton inside this module.
    """
    from local.storage.registry_provider import set_registry, reset_registry
    set_registry(reg)
    yield
    reset_registry()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _add(reg, lid="lecture_abc", display="Test Package", status="READY"):
    reg.add_lecture(
        lecture_id=lid,
        title="Test Title",
        display_name=display,
        chunk_count=10,
        segment_count=5,
        package_path="/tmp/fake_pkg",
        status=status,
    )


# ---------------------------------------------------------------------------
# 1. Provider contract
# ---------------------------------------------------------------------------

class TestRegistryProvider:

    def test_get_registry_returns_injected_instance(self, reg):
        from local.storage.registry_provider import get_registry
        assert get_registry() is reg

    def test_reset_registry_restores_default(self):
        from local.storage.registry_provider import set_registry, reset_registry, get_registry
        from local.storage.lecture_registry import LectureRegistry, registry as prod_registry
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as td:
            isolated = LectureRegistry(registry_file=pathlib.Path(td) / "r.json")
            set_registry(isolated)
            assert get_registry() is isolated
            reset_registry()
            # After reset, default (production singleton) should be returned.
            assert get_registry() is prod_registry

    def test_get_registry_default_is_production(self):
        """After reset, default registry points at production data/lecture_registry.json."""
        from local.storage.registry_provider import reset_registry, get_registry
        from local.storage.lecture_registry import registry as prod_registry
        reset_registry()
        try:
            assert get_registry() is prod_registry
        finally:
            # Re-inject so autouse fixture teardown works correctly.
            from local.storage.registry_provider import set_registry
            set_registry(None)  # will be overwritten by next test's autouse

    def test_production_registry_untouched(self, reg):
        """Writing to the injected registry must NOT modify the production file."""
        from local.storage.lecture_registry import REGISTRY_FILE
        prod_data_before = REGISTRY_FILE.read_text() if REGISTRY_FILE.exists() else None

        _add(reg)

        prod_data_after = REGISTRY_FILE.read_text() if REGISTRY_FILE.exists() else None
        assert prod_data_before == prod_data_after, (
            "Production registry was modified by a test-isolated write! "
            "The set_registry() injection is broken."
        )


# ---------------------------------------------------------------------------
# 2. CRUD lifecycle
# ---------------------------------------------------------------------------

class TestRegistryCRUD:

    def test_empty_registry_returns_empty_list(self, reg):
        assert reg.list_lectures() == []

    def test_add_and_retrieve(self, reg):
        _add(reg)
        lec = reg.get_lecture("lecture_abc")
        assert lec is not None
        assert lec["display_name"] == "Test Package"
        assert lec["status"] == "READY"

    def test_add_persists_to_disk(self, reg, tmp_path):
        _add(reg)
        raw = json.loads((tmp_path / "registry.json").read_text())
        assert "lecture_abc" in raw["lectures"]

    def test_update_status(self, reg):
        _add(reg, status="UPLOADING")
        reg.update_status("lecture_abc", "READY")
        assert reg.get_lecture("lecture_abc")["status"] == "READY"

    def test_update_metadata(self, reg):
        _add(reg)
        reg.update_metadata("lecture_abc", speaker="Prof. Smith", language="en")
        lec = reg.get_lecture("lecture_abc")
        assert lec["speaker"] == "Prof. Smith"
        assert lec["language"] == "en"

    def test_remove_lecture(self, reg):
        _add(reg)
        assert reg.get_lecture("lecture_abc") is not None
        reg.remove_lecture("lecture_abc")
        assert reg.get_lecture("lecture_abc") is None

    def test_remove_nonexistent_is_noop(self, reg):
        reg.remove_lecture("nonexistent_id")  # must not raise

    def test_list_returns_all(self, reg):
        _add(reg, "lecture_a", "Package A")
        _add(reg, "lecture_b", "Package B")
        ids = {lec["lecture_id"] for lec in reg.list_lectures()}
        assert ids == {"lecture_a", "lecture_b"}

    def test_display_name_defaults_to_title(self, reg):
        reg.add_lecture(
            lecture_id="lecture_x",
            title="Auto Title",
            chunk_count=0, segment_count=0,
            package_path="/tmp/x",
        )
        lec = reg.get_lecture("lecture_x")
        assert lec["display_name"] == "Auto Title"

    def test_get_nonexistent_returns_none(self, reg):
        assert reg.get_lecture("does_not_exist") is None


# ---------------------------------------------------------------------------
# 3. Active-lecture lifecycle
# ---------------------------------------------------------------------------

class TestActiveLecture:

    def test_set_and_get_active(self, reg):
        _add(reg)
        reg.set_active_lecture("lecture_abc")
        assert reg.get_active_lecture_id() == "lecture_abc"

    def test_active_id_none_on_fresh_registry(self, reg):
        assert reg.get_active_lecture_id() is None

    def test_active_persisted_on_disk(self, reg, tmp_path):
        _add(reg)
        reg.set_active_lecture("lecture_abc")
        raw = json.loads((tmp_path / "registry.json").read_text())
        assert raw.get("last_active_lecture_id") == "lecture_abc"


# ---------------------------------------------------------------------------
# 4. Registry recovery from corrupt file
# ---------------------------------------------------------------------------

class TestRegistryRecovery:

    def test_corrupt_json_starts_fresh(self, tmp_path):
        from local.storage.lecture_registry import LectureRegistry
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ this is: not valid json }")
        reg = LectureRegistry(registry_file=bad_file)
        assert reg.list_lectures() == []

    def test_missing_file_starts_fresh(self, tmp_path):
        from local.storage.lecture_registry import LectureRegistry
        reg = LectureRegistry(registry_file=tmp_path / "nonexistent.json")
        assert reg.list_lectures() == []

    def test_old_format_missing_lectures_key_is_upgraded(self, tmp_path):
        from local.storage.lecture_registry import LectureRegistry
        old = tmp_path / "old.json"
        old.write_text(json.dumps({"last_active_lecture_id": None}))
        reg = LectureRegistry(registry_file=old)
        assert reg.list_lectures() == []  # lectures key defaulted to {}


# ---------------------------------------------------------------------------
# 5. Consistency audit
# ---------------------------------------------------------------------------

class TestConsistencyAudit:

    def test_missing_package_dir_marked_failed(self, reg, tmp_path):
        # Add a package whose path doesn't exist.
        reg.add_lecture(
            lecture_id="lecture_missing",
            title="Gone",
            chunk_count=0, segment_count=0,
            package_path=str(tmp_path / "does_not_exist"),
            status="READY",
        )
        report = reg.audit(packages_dir=tmp_path)
        assert "lecture_missing" in report["missing_dirs"]
        # Status should have been updated to FAILED.
        assert reg.get_lecture("lecture_missing")["status"] == "FAILED"

    def test_valid_package_not_in_missing(self, reg, tmp_path):
        pkg_dir = tmp_path / "lecture_good"
        pkg_dir.mkdir()
        reg.add_lecture(
            lecture_id="lecture_good",
            title="Good",
            chunk_count=0, segment_count=0,
            package_path=str(pkg_dir),
            status="READY",
        )
        report = reg.audit(packages_dir=tmp_path)
        assert "lecture_good" not in report["missing_dirs"]

    def test_orphaned_dir_detected(self, reg, tmp_path):
        # Create a directory that is not in the registry.
        orphan = tmp_path / "lecture_orphan"
        orphan.mkdir()
        report = reg.audit(packages_dir=tmp_path)
        assert "lecture_orphan" in report["orphaned_dirs"]

    def test_duplicate_display_name_detected(self, reg, tmp_path):
        pkg_a = tmp_path / "lecture_a"; pkg_a.mkdir()
        pkg_b = tmp_path / "lecture_b"; pkg_b.mkdir()
        reg.add_lecture("lecture_a", "OS", 0, 0, str(pkg_a), display_name="Operating Systems")
        reg.add_lecture("lecture_b", "OS2", 0, 0, str(pkg_b), display_name="Operating Systems")
        report = reg.audit(packages_dir=tmp_path)
        assert "operating systems" in report["duplicate_names"]

    def test_no_issues_returns_empty_report(self, reg, tmp_path):
        pkg = tmp_path / "lecture_clean"; pkg.mkdir()
        reg.add_lecture("lecture_clean", "Clean", 0, 0, str(pkg))
        report = reg.audit(packages_dir=tmp_path)
        assert report["missing_dirs"] == []
        assert report["orphaned_dirs"] == []
        assert report["duplicate_names"] == []

    def test_audit_on_nonexistent_packages_dir_is_safe(self, reg):
        report = reg.audit(packages_dir=Path("/nonexistent_dir_xyz"))
        assert isinstance(report, dict)  # must not raise
