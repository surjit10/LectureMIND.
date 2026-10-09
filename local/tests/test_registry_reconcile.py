from pathlib import Path

from local.storage.lecture_registry import LectureRegistry
from scripts.reconcile_registry import reconcile


def test_reconcile_is_dry_run_by_default(tmp_path: Path):
    registry = LectureRegistry(tmp_path / "registry.json")
    package = tmp_path / "lecture_ok"
    package.mkdir()
    (package / "manifest.json").write_text("{}")
    registry.add_lecture("lecture_ok", "OK", 0, 0, str(package))

    report = reconcile(registry)
    assert report["actions"][0]["action"] == "record_content_hash"
    assert registry.get_lecture("lecture_ok")["package_sha256"] == ""


def test_reconcile_apply_never_deletes_and_marks_zip_failed(tmp_path: Path):
    registry = LectureRegistry(tmp_path / "registry.json")
    artifact = tmp_path / "lecture_zip.zip"
    artifact.write_bytes(b"artifact")
    registry.add_lecture("lecture_zip", "ZIP", 0, 0, str(artifact))

    report = reconcile(registry, apply=True)
    assert report["destructive_actions"] == 0
    assert artifact.exists()
    assert registry.get_lecture("lecture_zip")["status"] == "FAILED"
