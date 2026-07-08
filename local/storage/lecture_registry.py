import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import datetime

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REGISTRY_FILE = DATA_DIR / "lecture_registry.json"


class LectureRegistry:
    """
    Persistent store of imported knowledge-package metadata.

    Each instance is bound to a single JSON file on disk (``registry_file``).
    The production singleton in this module points at
    ``data/lecture_registry.json``.  Tests create their own instance pointing
    at a ``tmp_path``-backed file — no monkey-patching required.

    Consistency
    -----------
    Call :meth:`audit` on startup to detect and repair common issues:

    * Registry entry whose ``package_path`` directory no longer exists on disk
      → marked FAILED and logged.
    * Orphaned package directory with no registry entry → logged (not auto-imported;
      a corrupt partial import could be dangerous).
    * Duplicate ``display_name`` values → logged as a warning (not an error;
      users may legitimately give two packages the same display name).
    """

    def __init__(self, registry_file: str | Path = REGISTRY_FILE):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, object] = {"lectures": {}}
        self._load()
        logger.info(
            "[registry] Initialized — file=%s, packages=%d",
            self.registry_file,
            len(self._registry.get("lectures", {})),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # Upgrade: ensure top-level keys exist for backward compat.
                if "lectures" not in loaded:
                    loaded["lectures"] = {}
                self._registry = loaded
            except json.JSONDecodeError:
                logger.warning(
                    "[registry] Failed to decode registry file — starting fresh: %s",
                    self.registry_file,
                )
                self._registry = {"lectures": {}}
        else:
            logger.info(
                "[registry] No registry file found — starting with empty registry: %s",
                self.registry_file,
            )
            self._registry = {"lectures": {}}

    def _save(self) -> None:
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_lecture(
        self,
        lecture_id: str,
        title: str,
        chunk_count: int,
        segment_count: int,
        package_path: str,
        status: str = "READY",
        display_name: str = "",
        duration: float = 0.0,
        # V7 optional metadata
        course_name: str = "",
        speaker: str = "",
        description: str = "",
        language: str = "",
        pipeline_version: str = "",
        package_version: str = "",
    ) -> None:
        self._registry["lectures"][lecture_id] = {
            "lecture_id": lecture_id,
            "title": title,
            "display_name": display_name or title,
            "duration": duration,
            "status": status,
            "chunk_count": chunk_count,
            "segment_count": segment_count,
            "package_path": str(package_path),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            # V7 optional metadata
            "course_name": course_name,
            "speaker": speaker,
            "description": description,
            "language": language,
            "pipeline_version": pipeline_version,
            "package_version": package_version,
        }
        self._save()
        logger.info("[registry] Package registered: %s ('%s')", lecture_id, display_name or title)

    def get_lecture(self, lecture_id: str) -> Optional[dict]:
        return self._registry["lectures"].get(lecture_id)

    def list_lectures(self) -> List[dict]:
        return list(self._registry["lectures"].values())

    def update_status(self, lecture_id: str, status: str) -> None:
        if lecture_id in self._registry["lectures"]:
            self._registry["lectures"][lecture_id]["status"] = status
            self._save()
            logger.info("[registry] Status updated: %s → %s", lecture_id, status)

    def update_metadata(self, lecture_id: str, **kwargs) -> None:
        """
        Update any metadata fields for an existing lecture entry.

        V7 helper — allows enriching a lecture entry after import without
        calling add_lecture() again.  Only the supplied kwargs are updated.
        Example:
            registry.update_metadata(lecture_id, speaker="Prof. Smith", language="en")
        """
        if lecture_id in self._registry["lectures"]:
            self._registry["lectures"][lecture_id].update(kwargs)
            self._save()

    def remove_lecture(self, lecture_id: str) -> None:
        if lecture_id in self._registry["lectures"]:
            display = self._registry["lectures"][lecture_id].get("display_name", lecture_id)
            del self._registry["lectures"][lecture_id]
            self._save()
            logger.info("[registry] Package removed: %s ('%s')", lecture_id, display)

    # ------------------------------------------------------------------
    # Active-lecture lifecycle
    # ------------------------------------------------------------------

    def set_active_lecture(self, lecture_id: str) -> None:
        """Persist the last-activated lecture so it can be restored on startup."""
        self._registry["last_active_lecture_id"] = lecture_id
        self._registry["last_opened"] = (
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        self._save()
        logger.info("[registry] Active lecture set: %s", lecture_id)

    def get_active_lecture_id(self) -> Optional[str]:
        """Return the persisted active lecture id, or None."""
        return self._registry.get("last_active_lecture_id")  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Consistency audit  (Step 4 / Step 5 from the architecture prompt)
    # ------------------------------------------------------------------

    def audit(self, packages_dir: Path | None = None) -> Dict[str, list]:
        """
        Perform a consistency check between the registry and the filesystem.

        Checks
        ------
        1. Registry entry exists but ``package_path`` directory is missing on disk.
           → Updates status to FAILED and logs a warning.
        2. Package directory exists on disk but has no registry entry.
           → Logged as an orphan (no auto-import; partial/corrupt imports are dangerous).
        3. Duplicate ``display_name`` values.
           → Logged as a warning.

        Parameters
        ----------
        packages_dir:
            The directory that contains package sub-directories.  Defaults to
            ``PROJECT_ROOT / "data" / "packages"``.  Pass an override when
            testing.

        Returns
        -------
        dict with keys ``missing_dirs``, ``orphaned_dirs``, ``duplicate_names``
        each holding a list of IDs / names for reporting.
        """
        if packages_dir is None:
            packages_dir = PROJECT_ROOT / "data" / "packages"

        report: Dict[str, list] = {
            "missing_dirs": [],
            "orphaned_dirs": [],
            "duplicate_names": [],
        }

        logger.info("[registry] Starting consistency audit (packages_dir=%s).", packages_dir)

        # ── Check 1: registry entries with missing directories ─────────────
        for lid, entry in list(self._registry["lectures"].items()):
            pkg_path = Path(entry.get("package_path", ""))
            if not pkg_path.exists() or not pkg_path.is_dir():
                logger.warning(
                    "[registry] Audit: package directory missing for '%s' at '%s'. "
                    "Marking as FAILED.",
                    lid, pkg_path,
                )
                self._registry["lectures"][lid]["status"] = "FAILED"
                report["missing_dirs"].append(lid)
        if report["missing_dirs"]:
            self._save()

        # ── Check 2: orphaned directories with no registry entry ───────────
        registered_ids = set(self._registry["lectures"].keys())
        if packages_dir.is_dir():
            for child in packages_dir.iterdir():
                if child.is_dir() and child.name.startswith("lecture_"):
                    if child.name not in registered_ids:
                        logger.warning(
                            "[registry] Audit: orphaned package directory '%s' "
                            "has no registry entry. Manual inspection required.",
                            child,
                        )
                        report["orphaned_dirs"].append(child.name)

        # ── Check 3: duplicate display names ──────────────────────────────
        seen_names: Dict[str, str] = {}  # name → first lecture_id
        for lid, entry in self._registry["lectures"].items():
            name = (entry.get("display_name") or "").strip().lower()
            if name in seen_names:
                logger.warning(
                    "[registry] Audit: duplicate display_name '%s' used by '%s' and '%s'.",
                    name, seen_names[name], lid,
                )
                report["duplicate_names"].append(name)
            else:
                seen_names[name] = lid

        total_issues = sum(len(v) for v in report.values())
        if total_issues == 0:
            logger.info("[registry] Audit complete — no issues found.")
        else:
            logger.warning(
                "[registry] Audit complete — %d issue(s) found: %s",
                total_issues, report,
            )
        return report


# ---------------------------------------------------------------------------
# Module-level default instance
# ---------------------------------------------------------------------------
# This singleton exists for backward compatibility with code that does:
#   from local.storage.lecture_registry import registry
#
# New code should prefer:
#   from local.storage.registry_provider import get_registry
#
# Tests MUST NOT import this singleton directly.  Instead use:
#   from local.storage.registry_provider import set_registry, reset_registry
# ---------------------------------------------------------------------------
registry = LectureRegistry()
