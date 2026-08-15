# serving/fastapi/routes/courses.py
# Feature 1 — Course Index API.
#
# Courses are metadata-only (local/storage/course_registry.py). The
# course-scoped query endpoint fans out to each member lecture's EXISTING
# lecture-scoped retrievers (isolation guards apply per call), reranks with
# the global singleton, and generates the answer through the existing
# answer_generator_node — so evidence gating, sources, and language pinning
# behave exactly as in the single-lecture path.

import logging

from fastapi import APIRouter, HTTPException

from local.storage.course_registry import get_course_registry
from retrieval.course_retriever import retrieve_course_context
from retrieval.reranker.rerank_service import rerank
from agent.langgraph.nodes.answer_generator import answer_generator_node
from agent.dspy.planner import QueryPlanner
from serving.fastapi.schemas import (
    CourseAddLectureRequest,
    CourseCreateRequest,
    CourseInfo,
    CourseQueryRequest,
    CourseQueryResponse,
    CourseUpdateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _to_info(course: dict) -> CourseInfo:
    return CourseInfo(
        course_id=course["course_id"],
        name=course["name"],
        description=course.get("description", ""),
        lecture_ids=list(course.get("lecture_ids", [])),
        lecture_count=len(course.get("lecture_ids", [])),
        created_at=course.get("created_at", ""),
    )


@router.get("/courses", response_model=list[CourseInfo])
async def list_courses():
    """List all courses (metadata only)."""
    return [_to_info(c) for c in get_course_registry().list_courses()]


@router.post("/courses", response_model=CourseInfo, status_code=201)
async def create_course(request: CourseCreateRequest):
    """Create a course."""
    try:
        course = get_course_registry().create_course(request.name, request.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _to_info(course)


@router.get("/courses/{course_id}", response_model=CourseInfo)
async def get_course(course_id: str):
    """Get a single course."""
    course = get_course_registry().get_course(course_id)
    if course is None:
        raise HTTPException(status_code=404, detail="Course not found.")
    return _to_info(course)


@router.patch("/courses/{course_id}", response_model=CourseInfo)
async def update_course(course_id: str, request: CourseUpdateRequest):
    """Rename / re-describe a course."""
    try:
        course = get_course_registry().update_course(course_id, request.name, request.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if course is None:
        raise HTTPException(status_code=404, detail="Course not found.")
    return _to_info(course)


@router.delete("/courses/{course_id}", status_code=204)
async def delete_course(course_id: str):
    """Delete a course. Lectures themselves are untouched."""
    if not get_course_registry().delete_course(course_id):
        raise HTTPException(status_code=404, detail="Course not found.")
    return None


@router.post("/courses/{course_id}/lectures", response_model=CourseInfo)
async def add_lecture(course_id: str, request: CourseAddLectureRequest):
    """Add a READY lecture to a course."""
    try:
        course = get_course_registry().add_lecture(course_id, request.lecture_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _to_info(course)


@router.delete("/courses/{course_id}/lectures/{lecture_id}", status_code=204)
async def remove_lecture(course_id: str, lecture_id: str):
    """Remove a lecture from a course."""
    try:
        course = get_course_registry().remove_lecture(course_id, lecture_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if course is None:
        raise HTTPException(status_code=404, detail="Course not found.")
    return None


@router.post("/courses/query", response_model=CourseQueryResponse)
async def course_query(request: CourseQueryRequest):
    """
    Query across every lecture in a course.

    Flow: fan-out (existing lecture-scoped retrievers) → global reranker →
    existing answer generator. Every retrieval call is lecture-scoped, so
    lecture isolation is preserved by the existing guards.
    """
    registry = get_course_registry()
    course = registry.get_course(request.course_id)
    if course is None:
        raise HTTPException(status_code=404, detail="Course not found.")
    if not course.get("lecture_ids"):
        raise HTTPException(status_code=400, detail="Course has no lectures. Add lectures first.")

    plan = None
    try:
        plan = QueryPlanner().plan_full(request.query)
    except Exception:
        plan = None
    is_lecture_wide = plan.is_lecture_wide if plan else False
    need_visual = plan.need_visual if plan else False

    ctx = retrieve_course_context(
        request.query,
        request.course_id,
        is_lecture_wide=is_lecture_wide,
    )
    if not ctx["vector_results"] and not ctx["graph_results"]:
        return CourseQueryResponse(
            course_id=request.course_id,
            answer="Insufficient evidence found in the course lectures.",
            lectures_used=ctx["lectures_used"],
            skipped_lectures=ctx["skipped_lectures"],
        )

    reranked, final_context = rerank(
        request.query,
        ctx["graph_results"],
        ctx["vector_results"],
        # Keep ContextBuilder in point mode: per-lecture timeline sampling
        # already covers each lecture when lecture-wide; merged-timeline
        # bucket sampling across lectures would distort coverage.
        is_lecture_wide=False,
        need_visual=need_visual,
    )

    state = {
        "query": request.query,
        "graph_results": ctx["graph_results"],
        "vector_results": ctx["vector_results"],
        "reranked_results": reranked,
        "final_context": final_context,
    }
    out = answer_generator_node(state)

    return CourseQueryResponse(
        course_id=request.course_id,
        answer=out.get("answer", "Insufficient evidence found in the course lectures."),
        sources=out.get("sources", []),
        graph_path=out.get("graph_path", []),
        lectures_used=ctx["lectures_used"],
        skipped_lectures=ctx["skipped_lectures"],
    )
