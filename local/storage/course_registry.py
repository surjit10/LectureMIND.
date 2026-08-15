# local/storage/course_registry.py
# Feature 1 — Course Index (lightweight course layer).
#
# Courses are PURE METADATA. A course stores a name, description, and a list
# of lecture_ids. It never writes to Qdrant or Neo4j, never merges lecture
# graphs, and never touches another lecture's entities. Course queries fan
# out to the existing lecture-scoped retrievers (each under its own
# lecture_id guard) — see retrieval/course_retriever.py.
#
# Isolation invariant: lecture graphs remain byte-for-byte independent. This
# registry is the ONLY place course membership lives.

import datetime
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from local.storage.lecture_registry import DATA_DIR

logger = logging.getLogger(__name__)

COURSES_FILE = DATA_DIR / "courses.json"


class CourseRegistry:
    """
    Persistent store of course metadata (JSON on disk).

    Courses point at lecture_ids by value. Lecture isolation is preserved:
    deleting a lecture does not delete the course — the course simply keeps
    the id (queries skip missing lectures with a warning).
    """

    def __init__(self, courses_file: str | Path = COURSES_FILE):
        self.courses_file = Path(courses_file)
        self.courses_file.parent.mkdir(parents=True, exist_ok=True)
        self._courses: Dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.courses_file.exists():
            try:
                self._courses = json.loads(self.courses_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("[courses] Failed to decode %s — starting fresh.", self.courses_file)
                self._courses = {}
        if not isinstance(self._courses, dict):
            self._courses = {}

    def _save(self) -> None:
        self.courses_file.write_text(json.dumps(self._courses, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_course(self, name: str, description: str = "") -> dict:
        """Create a course and persist it. Returns the stored course dict."""
        name = (name or "").strip()
        if not name:
            raise ValueError("Course name must not be empty.")
        course_id = f"course_{uuid.uuid4().hex[:8]}"
        course = {
            "course_id": course_id,
            "name": name,
            "description": description or "",
            "lecture_ids": [],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._courses[course_id] = course
        self._save()
        logger.info("[courses] Created course %s ('%s')", course_id, name)
        return dict(course)

    def get_course(self, course_id: str) -> Optional[dict]:
        course = self._courses.get(course_id)
        return dict(course) if course else None

    def list_courses(self) -> List[dict]:
        return [dict(c) for c in self._courses.values()]

    def update_course(self, course_id: str, name: Optional[str] = None, description: Optional[str] = None) -> Optional[dict]:
        course = self._courses.get(course_id)
        if course is None:
            return None
        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("Course name must not be empty.")
            course["name"] = name
        if description is not None:
            course["description"] = description
        self._save()
        return dict(course)

    def delete_course(self, course_id: str) -> bool:
        if course_id in self._courses:
            del self._courses[course_id]
            self._save()
            logger.info("[courses] Deleted course %s", course_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------

    def add_lecture(self, course_id: str, lecture_id: str) -> Optional[dict]:
        """
        Add a lecture to a course. The lecture must exist and be READY in the
        lecture registry (prevents courses pointing at garbage).
        """
        course = self._courses.get(course_id)
        if course is None:
            raise KeyError(f"Course not found: {course_id}")

        from local.storage.registry_provider import get_registry
        lecture = get_registry().get_lecture(lecture_id)
        if not lecture or lecture.get("status") != "READY":
            raise ValueError(f"Lecture {lecture_id} is not imported/READY — cannot add to course.")

        if lecture_id not in course["lecture_ids"]:
            course["lecture_ids"].append(lecture_id)
            self._save()
        return dict(course)

    def remove_lecture(self, course_id: str, lecture_id: str) -> Optional[dict]:
        course = self._courses.get(course_id)
        if course is None:
            raise KeyError(f"Course not found: {course_id}")
        if lecture_id in course["lecture_ids"]:
            course["lecture_ids"].remove(lecture_id)
            self._save()
        return dict(course)

    def lecture_ids(self, course_id: str) -> List[str]:
        course = self._courses.get(course_id)
        return list(course["lecture_ids"]) if course else []

    # ------------------------------------------------------------------
    # One-time backfill from lecture registry metadata
    # ------------------------------------------------------------------

    def backfill_from_lecture_registry(self) -> int:
        """
        Group READY lectures by their stored course_name into courses.
        Idempotent: a lecture already in a course with that name is skipped.
        Returns the number of courses created.
        """
        from local.storage.registry_provider import get_registry

        created = 0
        for lecture in get_registry().list_lectures():
            name = (lecture.get("course_name") or "").strip()
            if not name or lecture.get("status") != "READY":
                continue
            # Reuse an existing course with the same name when present.
            course_id = next(
                (cid for cid, c in self._courses.items() if c["name"].lower() == name.lower()),
                None,
            )
            if course_id is None:
                course = self.create_course(name)
                course_id = course["course_id"]
                created += 1
            if lecture["lecture_id"] not in self._courses[course_id]["lecture_ids"]:
                self._courses[course_id]["lecture_ids"].append(lecture["lecture_id"])
                self._save()
        logger.info("[courses] Backfill complete — %d course(s) created.", created)
        return created


# ---------------------------------------------------------------------------
# Provider (mirrors registry_provider.py — test injection without monkeypatching)
# ---------------------------------------------------------------------------

_course_registry_instance: Optional[CourseRegistry] = None
_default_course_registry: Optional[CourseRegistry] = None


def get_course_registry() -> CourseRegistry:
    global _course_registry_instance, _default_course_registry
    if _course_registry_instance is not None:
        return _course_registry_instance
    if _default_course_registry is None:
        _default_course_registry = CourseRegistry()
    return _default_course_registry


def set_course_registry(instance: CourseRegistry) -> None:
    global _course_registry_instance
    _course_registry_instance = instance


def reset_course_registry() -> None:
    global _course_registry_instance
    _course_registry_instance = None
