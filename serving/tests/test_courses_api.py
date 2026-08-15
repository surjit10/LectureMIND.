# serving/tests/test_courses_api.py
# Feature 1 — Courses API. Uses isolated registries (no production data).

import pytest
from fastapi.testclient import TestClient

from local.storage.course_registry import CourseRegistry, set_course_registry, reset_course_registry
from local.storage.lecture_registry import LectureRegistry
from local.storage.registry_provider import set_registry, reset_registry
from serving.fastapi.app import app


@pytest.fixture
def client(tmp_path):
    # The app startup audit marks lectures whose package dir is missing as
    # FAILED — create the dir so the fake lecture stays READY.
    (tmp_path / "a").mkdir()
    lreg = LectureRegistry(registry_file=tmp_path / "l.json")
    lreg.add_lecture("lecture_aaa", "DB1", 10, 2, str(tmp_path / "a"), status="READY")
    set_registry(lreg)
    set_course_registry(CourseRegistry(courses_file=tmp_path / "c.json"))
    with TestClient(app) as c:
        yield c
    reset_course_registry()
    reset_registry()


class TestCourseCrud:
    def test_create_list_get(self, client):
        r = client.post("/courses", json={"name": "Databases", "description": "DB course"})
        assert r.status_code == 201
        course = r.json()
        assert course["name"] == "Databases"
        assert course["lecture_count"] == 0

        r = client.get("/courses")
        assert r.status_code == 200
        assert len(r.json()) == 1

        r = client.get(f"/courses/{course['course_id']}")
        assert r.status_code == 200
        assert r.json()["name"] == "Databases"

    def test_empty_name_rejected(self, client):
        r = client.post("/courses", json={"name": "  "})
        assert r.status_code == 400

    def test_membership_flow(self, client):
        course = client.post("/courses", json={"name": "Databases"}).json()
        r = client.post(
            f"/courses/{course['course_id']}/lectures",
            json={"lecture_id": "lecture_aaa"},
        )
        assert r.status_code == 200
        assert r.json()["lecture_ids"] == ["lecture_aaa"]

        # Unknown lecture rejected.
        r = client.post(
            f"/courses/{course['course_id']}/lectures",
            json={"lecture_id": "lecture_nope"},
        )
        assert r.status_code == 400

        r = client.delete(f"/courses/{course['course_id']}/lectures/lecture_aaa")
        assert r.status_code == 204

    def test_delete_course(self, client):
        course = client.post("/courses", json={"name": "Databases"}).json()
        assert client.delete(f"/courses/{course['course_id']}").status_code == 204
        assert client.get(f"/courses/{course['course_id']}").status_code == 404


class TestCourseQuery:
    def test_unknown_course_404(self, client):
        r = client.post("/courses/query", json={"course_id": "course_x", "query": "q"})
        assert r.status_code == 404

    def test_empty_course_400(self, client):
        course = client.post("/courses", json={"name": "Empty"}).json()
        r = client.post("/courses/query", json={"course_id": course["course_id"], "query": "q"})
        assert r.status_code == 400

    def test_course_query_insufficient_when_no_retrieval(self, client, monkeypatch):
        course = client.post("/courses", json={"name": "DB"}).json()
        client.post(f"/courses/{course['course_id']}/lectures", json={"lecture_id": "lecture_aaa"})

        def fake_fanout(*args, **kwargs):
            return {"vector_results": [], "graph_results": [], "lectures_used": ["lecture_aaa"], "skipped_lectures": []}

        from serving.fastapi.routes import courses as courses_route
        monkeypatch.setattr(courses_route, "retrieve_course_context", fake_fanout)

        r = client.post("/courses/query", json={"course_id": course["course_id"], "query": "q"})
        assert r.status_code == 200
        assert "Insufficient evidence" in r.json()["answer"]
        assert r.json()["lectures_used"] == ["lecture_aaa"]
