# retrieval/tests/test_e2b_retrieval_pool.py
# Regression tests for the wide retrieval candidate pool:
#   1. Normal (non lecture-wide) retrieval candidate pool widened 5 -> 15
#      (the reranker can only recover candidates that enter its pool).
#   2. Query-time filter of transcript-only filler chunks (< 15 chars,
#      no OCR/visual content) so "OK?" / "Okay." stop stealing slots.
#
# Contracts preserved (asserted here or by untouched existing tests):
#   - lecture-wide top_k / budget unchanged
#   - lecture_id filter still required + applied
#   - multimodal chunks (short/empty transcript + OCR/visual) never filtered
#   - legitimate short facts (>= 15 chars) never filtered
#   - chunk_id / timestamp / payload structure preserved
#   - the final context budget is unchanged

import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.dspy.planner import QueryPlanner, NORMAL_TOP_K, LECTURE_WIDE_TOP_K
from retrieval.vector_retriever.qdrant_retriever import (
    retrieve_vectors,
    RETRIEVAL_FETCH_SLACK,
    MIN_TRANSCRIPT_CHARS,
)
from agent.langgraph.nodes.vector_retriever import vector_retriever_node


def _payload(chunk_id, transcript="", ocr_text="", visual_context="", timestamp=10.0, lecture_id="lec_001", segment_id="s_001"):
    return {
        "chunk_id": chunk_id,
        "transcript": transcript,
        "ocr_text": ocr_text,
        "visual_context": visual_context,
        "timestamp": timestamp,
        "segment_id": segment_id,
        "lecture_id": lecture_id,
    }


def _hit(payload, score=0.8):
    return SimpleNamespace(payload=payload, score=score)


def _mock_client(points):
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.points = points
    mock.query_points.return_value = mock_response
    return mock


def _mock_model(dim=1024):
    mock = MagicMock()
    mock.encode.return_value = np.random.randn(1, dim).astype(np.float32)
    return mock


class TestCandidatePoolSize:

    def test_normal_top_k_is_15(self):
        """Normal (non lecture-wide) queries now retrieve a pool of 15."""
        plan = QueryPlanner().plan_full("What is normalization?")
        assert plan.is_lecture_wide is False
        assert plan.top_k == NORMAL_TOP_K == 15

    def test_lecture_wide_top_k_unchanged(self):
        """Lecture-wide retrieval keeps its existing top_k and budget."""
        plan = QueryPlanner().plan_full("Summarize the lecture")
        assert plan.is_lecture_wide is True
        assert plan.top_k == LECTURE_WIDE_TOP_K == 15
        assert plan.context_budget == 6000

    def test_normal_context_budget_unchanged(self):
        """Widening the pool must not widen the LLM context budget."""
        plan = QueryPlanner().plan_full("What is a primary key?")
        assert plan.context_budget == 4000

    def test_vector_node_passes_expanded_top_k(self, monkeypatch):
        """vector_retriever_node hands the planner's top_k to retrieve_vectors."""
        captured = {}
        def fake_retrieve(query, qdrant_client=None, embedding_model=None,
                          top_k=5, lecture_id=None, **kwargs):
            captured["top_k"] = top_k
            captured["lecture_id"] = lecture_id
            return []
        monkeypatch.setattr(
            "agent.langgraph.nodes.vector_retriever.retrieve_vectors",
            fake_retrieve,
        )
        state = {"query": "What is a primary key?", "lecture_id": "lec_001"}
        vector_retriever_node(state, qdrant_client=MagicMock(), embedding_model=_mock_model())
        assert captured["top_k"] == NORMAL_TOP_K == 15
        assert captured["lecture_id"] == "lec_001"

    def test_retrieval_requests_slack_to_cover_filtering(self):
        """Qdrant is asked for top_k + slack so filtering never shrinks the pool."""
        filler = _hit(_payload("c_filler", transcript="OK?"), score=0.9)
        valid = _hit(_payload("c_001", transcript="A real sentence about kernels."), score=0.8)
        client = _mock_client([filler, valid])
        results = retrieve_vectors(
            "kernel question", qdrant_client=client, embedding_model=_mock_model(),
            top_k=5, lecture_id="lec_001",
        )
        call = client.query_points.call_args
        assert call.kwargs["limit"] == 5 + RETRIEVAL_FETCH_SLACK
        # Externally requested top_k semantics preserved: up to 5 valid results.
        assert len(results) == 1


class TestFillerFilter:

    def test_filler_filtered(self):
        """Pure transcript filler ("OK?") is excluded from candidates."""
        filler = _hit(_payload("c_filler", transcript="OK?"), score=0.9)
        valid = _hit(_payload("c_001", transcript="A real sentence about kernels."), score=0.8)
        results = retrieve_vectors(
            "kernel question", qdrant_client=_mock_client([filler, valid]),
            embedding_model=_mock_model(), top_k=5, lecture_id="lec_001",
        )
        ids = [r["chunk_id"] for r in results]
        assert "c_filler" not in ids
        assert "c_001" in ids

    def test_short_filler_variants_filtered(self):
        """'Okay.' / 'OK?' / 'Yeah.' are filler and filtered."""
        for filler_text in ("Okay.", "OK?", "Yeah.", "OK"):
            filler = _hit(_payload("c_f", transcript=filler_text), score=0.9)
            results = retrieve_vectors(
                "test query", qdrant_client=_mock_client([filler]),
                embedding_model=_mock_model(), top_k=5, lecture_id="lec_001",
            )
            assert results == [], f"expected '{filler_text}' to be filtered"

    def test_legitimate_short_fact_kept(self):
        """A short factual chunk (>= 15 chars) must remain eligible."""
        fact = _hit(_payload("c_428", transcript="The class has a limit of 428."), score=0.8)
        results = retrieve_vectors(
            "enrollment limit", qdrant_client=_mock_client([fact]),
            embedding_model=_mock_model(), top_k=5, lecture_id="lec_001",
        )
        assert [r["chunk_id"] for r in results] == ["c_428"]

    def test_threshold_is_exactly_15(self):
        """13 chars is filler; 15+ chars is kept (boundary of MIN_TRANSCRIPT_CHARS)."""
        short = _hit(_payload("c_short", transcript="BFS uses queue"), score=0.9)  # 13 chars
        boundary = _hit(_payload("c_bound", transcript="BFS uses a queue!"), score=0.8)  # 17 chars
        results = retrieve_vectors(
            "bfs", qdrant_client=_mock_client([short, boundary]),
            embedding_model=_mock_model(), top_k=5, lecture_id="lec_001",
        )
        ids = [r["chunk_id"] for r in results]
        assert "c_short" not in ids
        assert "c_bound" in ids
        assert MIN_TRANSCRIPT_CHARS == 15

    def test_multimodal_chunk_never_filtered(self):
        """Short/empty transcript with OCR or visual content is valid evidence."""
        visual = _hit(_payload("c_vis", transcript="", visual_context="### Slide diagram of OS layers"), score=0.8)
        ocr = _hit(_payload("c_ocr", transcript="OK?", ocr_text="Virtual Memory"), score=0.8)
        results = retrieve_vectors(
            "os diagram", qdrant_client=_mock_client([visual, ocr]),
            embedding_model=_mock_model(), top_k=5, lecture_id="lec_001",
        )
        ids = [r["chunk_id"] for r in results]
        assert "c_vis" in ids
        assert "c_ocr" in ids

    def test_chunk_id_and_timestamp_preserved(self):
        """Filtered results keep chunk_id and timestamp in the payload."""
        valid = _hit(_payload("c_001", transcript="A real sentence about kernels.", timestamp=42.5), score=0.8)
        results = retrieve_vectors(
            "kernel question", qdrant_client=_mock_client([valid]),
            embedding_model=_mock_model(), top_k=5, lecture_id="lec_001",
        )
        assert results[0]["chunk_id"] == "c_001"
        assert results[0]["payload"]["timestamp"] == 42.5
        assert results[0]["payload"]["chunk_id"] == "c_001"
        assert "payload" in results[0]

    def test_lecture_id_filter_still_applied(self):
        """The Qdrant query keeps the lecture_id filter (isolation contract)."""
        valid = _hit(_payload("c_001", transcript="A real sentence about kernels."), score=0.8)
        client = _mock_client([valid])
        retrieve_vectors(
            "kernel question", qdrant_client=client, embedding_model=_mock_model(),
            top_k=5, lecture_id="lec_001",
        )
        qfilter = client.query_points.call_args.kwargs["query_filter"]
        assert qfilter is not None
        assert qfilter.must[0].key == "lecture_id"
        assert qfilter.must[0].match.value == "lec_001"

    def test_missing_lecture_id_still_raises(self):
        """Retrieval without a lecture_id must still fail loudly."""
        with pytest.raises(ValueError, match="lecture_id is required"):
            retrieve_vectors(
                "test", qdrant_client=_mock_client([]), embedding_model=_mock_model(),
            )
