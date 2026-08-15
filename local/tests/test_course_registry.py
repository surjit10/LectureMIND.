# local/tests/test_course_registry.py
# Feature 1 — Course Registry (metadata-only; no graph/vector writes).

import pytest

from local.storage.course_registry import CourseRegistry
from local.storage.lecture_registry import LectureRegistry
from local.storage.registry_provider import set_registry, reset_registry


@pytest.fixture
def lecture_registry(tmp_path):
    reg = LectureRegistry(registry_file=tmp_path / "lecture_registry.json")
    reg.add_lecture(
        lecture_id="lecture_aaa",
        title="DB 1",
        chunk_count=10,
        segment_count=2,
        package_path=str(tmp_path / "lecture_aaa"),
        status="READY",
        course_name="Databases",
    )
    reg.add_lecture(
        lecture_id="lecture_bbb",
        title="DB 2",
        chunk_count=10,
        segment_count=2,
        package_path=str(tmp_path / "lecture_bbb"),
        status="READY",
        course_name="Databases",
    )
    set_registry(reg)
    yield reg
    reset_registry()


class TestCourseCrud:
    def test_create_get_list_delete(self, tmp_path):
        reg = CourseRegistry(courses_file=tmp_path / "courses.json")
        course = reg.create_course("Databases", "DB lectures")
        assert course["name"] == "Databases"
        assert course["lecture_ids"] == []
        assert reg.get_course(course["course_id"])["course_id"] == course["course_id"]
        assert len(reg.list_courses()) == 1
        assert reg.delete_course(course["course_id"]) is True
        assert reg.get_course(course["course_id"]) is None

    def test_empty_name_rejected(self, tmp_path):
        reg = CourseRegistry(courses_file=tmp_path / "courses.json")
        with pytest.raises(ValueError):
            reg.create_course("   ")

    def test_persistence(self, tmp_path):
        path = tmp_path / "courses.json"
        reg = CourseRegistry(courses_file=path)
        reg.create_course("Databases")
        # New instance reads from disk.
        reg2 = CourseRegistry(courses_file=path)
        assert len(reg2.list_courses()) == 1


class TestMembership:
    def test_add_lecture_requires_ready_lecture(self, tmp_path, lecture_registry):
        reg = CourseRegistry(courses_file=tmp_path / "courses.json")
        course = reg.create_course("Databases")
        reg.add_lecture(course["course_id"], "lecture_aaa")
        assert reg.lecture_ids(course["course_id"]) == ["lecture_aaa"]
        # Duplicate add is idempotent.
        reg.add_lecture(course["course_id"], "lecture_aaa")
        assert reg.lecture_ids(course["course_id"]) == ["lecture_aaa"]

    def test_add_unknown_lecture_rejected(self, tmp_path, lecture_registry):
        reg = CourseRegistry(courses_file=tmp_path / "courses.json")
        course = reg.create_course("Databases")
        with pytest.raises(ValueError):
            reg.add_lecture(course["course_id"], "lecture_nope")

    def test_remove_lecture(self, tmp_path, lecture_registry):
        reg = CourseRegistry(courses_file=tmp_path / "courses.json")
        course = reg.create_course("Databases")
        reg.add_lecture(course["course_id"], "lecture_aaa")
        reg.remove_lecture(course["course_id"], "lecture_aaa")
        assert reg.lecture_ids(course["course_id"]) == []


class TestBackfill:
    def test_backfill_groups_by_course_name(self, tmp_path, lecture_registry):
        reg = CourseRegistry(courses_file=tmp_path / "courses.json")
        created = reg.backfill_from_lecture_registry()
        assert created == 1
        courses = reg.list_courses()
        assert courses[0]["name"] == "Databases"
        assert set(courses[0]["lecture_ids"]) == {"lecture_aaa", "lecture_bbb"}

    def test_backfill_idempotent(self, tmp_path, lecture_registry):
        reg = CourseRegistry(courses_file=tmp_path / "courses.json")
        reg.backfill_from_lecture_registry()
        reg.backfill_from_lecture_registry()
        assert len(reg.list_courses()) == 1
