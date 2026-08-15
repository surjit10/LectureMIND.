# retrieval/tests/test_course_retriever.py
# Feature 1 — Course fan-out retriever.
# Isolation checks: every underlying retrieval must be called with an explicit
# lecture_id; results dedupe on (lecture_id, chunk_id).

import pytest

from local.storage.course_registry import CourseRegistry
from local.storage.lecture_registry import LectureRegistry
from local.storage.registry_provider import set_registry, reset_registry
from retrieval import course_retriever


@pytest.fixture
def two_lecture_course(tmp_path, monkeypatch):
    lreg = LectureRegistry(registry_file=tmp_path / "l.json")
    lreg.add_lecture("lecture_aaa", "A", 1, 1, str(tmp_path / "a"), status="READY")
    lreg.add_lecture("lecture_bbb", "B", 1, 1, str(tmp_path / "b"), status="READY")
    set_registry(lreg)

    creg = CourseRegistry(courses_file=tmp_path / "c.json")
    course = creg.create_course("Course")
    creg.add_lecture(course["course_id"], "lecture_aaa")
    creg.add_lecture(course["course_id"], "lecture_bbb")
    # Inject the test course registry via its provider.
    from local.storage import course_registry as cr_mod
    monkeypatch.setattr(cr_mod, "_default_course_registry", creg)

    yield creg, course
    reset_registry()


def _hit(chunk_id, lecture_id, score):
    return {"chunk_id": chunk_id, "score": score, "payload": {
        "chunk_id": chunk_id, "lecture_id": lecture_id, "transcript": "text",
    }}


class TestFanOut:
    def test_queries_each_lecture_with_its_own_id(self, two_lecture_course, monkeypatch):
        _, course = two_lecture_course
        calls = []

        def fake_vector(query, qdrant_client=None, embedding_model=None, top_k=5, lecture_id=None, **kw):
            calls.append(lecture_id)
            if lecture_id == "lecture_aaa":
                return [_hit("a1", "lecture_aaa", 0.9)]
            return [_hit("b1", "lecture_bbb", 0.7)]

        monkeypatch.setattr(course_retriever, "retrieve_vectors", fake_vector)
        monkeypatch.setattr(course_retriever, "retrieve_graph", lambda *a, **k: [])
        monkeypatch.setattr(course_retriever, "preload_embedding_model", lambda *a, **k: object())

        out = course_retriever.retrieve_course_context("q", course["course_id"])
        assert sorted(calls) == ["lecture_aaa", "lecture_bbb"]
        assert out["lectures_used"] == ["lecture_aaa", "lecture_bbb"]
        assert len(out["vector_results"]) == 2

    def test_dedup_same_chunk_id_across_lectures(self, two_lecture_course, monkeypatch):
        _, course = two_lecture_course

        def fake_vector(query, qdrant_client=None, embedding_model=None, top_k=5, lecture_id=None, **kw):
            # Same chunk_id string in BOTH lectures — must not collapse.
            return [_hit("dup_chunk", lecture_id, 0.8)]

        monkeypatch.setattr(course_retriever, "retrieve_vectors", fake_vector)
        monkeypatch.setattr(course_retriever, "retrieve_graph", lambda *a, **k: [])
        monkeypatch.setattr(course_retriever, "preload_embedding_model", lambda *a, **k: object())

        out = course_retriever.retrieve_course_context("q", course["course_id"])
        assert len(out["vector_results"]) == 2  # one per lecture, both kept

    def test_failed_lecture_is_skipped_not_fatal(self, two_lecture_course, monkeypatch):
        _, course = two_lecture_course

        def fake_vector(query, qdrant_client=None, embedding_model=None, top_k=5, lecture_id=None, **kw):
            if lecture_id == "lecture_bbb":
                raise RuntimeError("Qdrant down")
            return [_hit("a1", "lecture_aaa", 0.9)]

        monkeypatch.setattr(course_retriever, "retrieve_vectors", fake_vector)
        monkeypatch.setattr(course_retriever, "retrieve_graph", lambda *a, **k: [])
        monkeypatch.setattr(course_retriever, "preload_embedding_model", lambda *a, **k: object())

        out = course_retriever.retrieve_course_context("q", course["course_id"])
        assert "lecture_bbb" in out["skipped_lectures"]
        assert "lecture_bbb" not in out["lectures_used"]
        assert len(out["vector_results"]) == 1

    def test_unknown_course_raises(self, two_lecture_course):
        with pytest.raises(KeyError):
            course_retriever.retrieve_course_context("q", "course_nope")
