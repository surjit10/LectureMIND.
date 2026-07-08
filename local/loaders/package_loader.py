import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from local.storage.registry_provider import get_registry

logger = logging.getLogger(__name__)


def load_package(
    lecture_id: str,
    registry=None,  # optional explicit injection for tests
) -> Dict[str, Any]:
    """
    Load JSON files from a validated knowledge package into memory.

    Parameters
    ----------
    lecture_id:
        Registry key for the package to load.
    registry:
        Optional explicit registry instance.  When None (default),
        ``get_registry()`` is called, which returns the currently injected
        instance (production or test).

    Returns
    -------
    dict containing all JSON artifacts keyed by filename stem.

    Raises
    ------
    ValueError: if the package is not found in the registry or a required
        file cannot be read.
    """
    reg = registry or get_registry()
    lecture_info = reg.get_lecture(lecture_id)
    if not lecture_info:
        raise ValueError(f"Package '{lecture_id}' not found in registry.")

    package_dir = Path(lecture_info["package_path"])
    if not package_dir.is_dir():
        raise ValueError(
            f"Package directory for '{lecture_id}' does not exist: {package_dir}"
        )

    package: Dict[str, Any] = {
        "lecture_id": lecture_id,
        "package_dir": str(package_dir),
    }

    files_to_load = [
        "manifest.json",
        "segments.json",
        "entities.json",
        "relations.json",
    ]

    optional_files = [
        "metadata.json",
        "triplets.json"
    ]

    for filename in files_to_load + optional_files:
        file_path = package_dir / filename
        if filename in optional_files and not file_path.exists():
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                key_name = filename.split(".")[0]
                package[key_name] = json.load(f)
        except Exception as exc:
            logger.error(
                "[package_loader] Failed to load %s from package %s: %s",
                filename, lecture_id, exc,
            )
            raise ValueError(f"Corrupted package file: {filename}") from exc

    logger.info("[package_loader] Package loaded: %s", lecture_id)
    return package
