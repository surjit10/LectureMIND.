"""Create reproducible run metadata without requiring an external tracker."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _git(args: list[str], root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True,
            check=True, timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def create_run_manifest(
    run_type: str,
    parameters: Optional[Dict[str, Any]] = None,
    project_root: Optional[Path] = None,
    dependency_names: Iterable[str] = (
        "fastapi", "pydantic", "numpy", "qdrant-client", "neo4j",
        "sentence-transformers",
    ),
) -> Dict[str, Any]:
    """Return JSON-serializable code, environment, and parameter provenance."""
    root = project_root or Path(__file__).resolve().parent.parent
    status = _git(["status", "--porcelain"], root)
    dependencies: Dict[str, str] = {}
    for name in dependency_names:
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue

    return {
        "run_id": f"{run_type}_{uuid.uuid4().hex[:12]}",
        "run_type": run_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code": {
            "git_commit": _git(["rev-parse", "HEAD"], root),
            "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
            "dirty": bool(status) if status is not None else None,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": dependencies,
        },
        "parameters": parameters or {},
    }
