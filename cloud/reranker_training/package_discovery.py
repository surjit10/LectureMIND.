# cloud/reranker_training/package_discovery.py
#
# Pipeline B, Step 1 — discover lecture knowledge packages and extract ONLY
# their triplets.json. Everything else in a package is ignored.
#
# Sources are either:
#   * a knowledge-package ZIP (any *.zip; prefer *_knowledge_package.zip) — the
#     file is opened read-only, only the triplets.json member is read, the rest
#     of the archive is never extracted.
#   * an already-extracted package directory that contains triplets.json.
#
# Robustness contract (shared with dataset_builder):
#   * missing packages        → skipped, logged, job continues
#   * corrupted zips          → skipped, logged, job continues
#   * missing triplets.json   → skipped, logged, job continues
#   * malformed triplet rows  → dropped individually, valid rows kept
#
# Environment: Kaggle GPU notebook (Pipeline B). No dependency on Pipeline A.

import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schemas.triplet import RerankerTriplet

logger = logging.getLogger(__name__)

TRIPLETS_FILENAME = "triplets.json"
# Hidden dirs (Kaggle input mounts include "." entries) are never scanned.
HIDDEN_DIR_PREFIX = "."
# Hard cap on the triplets.json payload read from any single package. A
# malicious or runaway package must degrade to a skipped-source error, never
# an OOM that kills the whole training run.
MAX_TRIPLETS_BYTES = 200 * 1024 * 1024


@dataclass(frozen=True)
class PackageSource:
    """A discovered knowledge package: either a zip path or an extracted dir."""

    path: Path
    is_zip: bool

    @property
    def label(self) -> str:
        return f"{'zip' if self.is_zip else 'dir'}:{self.path}"

    @property
    def lecture_key(self) -> str:
        """Stable, human-readable lecture identifier for metadata/reporting.

        Strips the knowledge-package naming convention so metadata lists
        clean ids (e.g. "lecture1" instead of "lecture1_knowledge_package").
        """
        name = self.path.stem if self.is_zip else self.path.name
        suffix = "_knowledge_package"
        if name.endswith(suffix):
            name = name[: -len(suffix)]
        return name


@dataclass
class TripletExtraction:
    """Result of extracting triplets from one package source."""

    source: PackageSource
    triplets: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    n_dropped: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None


def discover_packages(
    packages_root: Path,
    exclude_dirs: Optional[List[str]] = None,
) -> List[PackageSource]:
    """
    Recursively discover knowledge packages under packages_root.

    Args:
        packages_root: Root to scan (e.g. /kaggle/input or a dataset mount).
        exclude_dirs: Directory names to skip entirely (e.g. "datasets" so the
            big Kaggle model datasets are never scanned). Defaults to
            ["datasets"].

    Returns:
        Sorted list of PackageSource (zips first, then extracted dirs).
    """
    exclude = set(exclude_dirs) if exclude_dirs is not None else {"datasets"}
    packages_root = Path(packages_root)

    if not packages_root.is_dir():
        logger.warning("Pipeline B: packages root not found: %s", packages_root)
        return []

    zips: List[Path] = []
    dirs: List[Path] = []

    for dirpath, dirnames, filenames in os_walk(packages_root):
        # Prune excluded and hidden directories in-place.
        dirnames[:] = [
            d for d in dirnames
            if d not in exclude and not d.startswith(HIDDEN_DIR_PREFIX)
        ]
        for name in filenames:
            if name.lower().endswith(".zip"):
                zips.append(Path(dirpath) / name)
        if TRIPLETS_FILENAME in filenames:
            dirs.append(Path(dirpath))

    sources: List[PackageSource] = []
    for zp in sorted(zips):
        sources.append(PackageSource(path=zp, is_zip=True))
    for d in sorted(dirs):
        # Skip dirs that are themselves inside a discovered zip-extraction tree
        # (harmless duplicate sources otherwise).
        sources.append(PackageSource(path=d, is_zip=False))

    logger.info(
        "Pipeline B: discovered %d package source(s) under %s "
        "(zips=%d, extracted dirs=%d).",
        len(sources), packages_root, len(zips), len(dirs),
    )
    return sources


def os_walk(root: Path):
    """Thin shim over os.walk for testability; yields (dirpath, dirnames, filenames)."""
    import os

    return os.walk(root)


def extract_triplets(source: PackageSource) -> TripletExtraction:
    """Extract + validate triplets.json from a single package source.

    Never raises for corrupt/missing data — those become `.error`.
    Only rows that pass the RerankerTriplet closed schema are kept.
    """
    if source.is_zip:
        return _extract_from_zip(source)
    return _extract_from_dir(source)


def _extract_from_zip(source: PackageSource) -> TripletExtraction:
    zip_path = Path(source.path)
    if not zipfile.is_zipfile(zip_path):
        return TripletExtraction(source=source, error="not a valid zip archive")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            # Fast path: only check the member list; read content only when present.
            if TRIPLETS_FILENAME not in names:
                return TripletExtraction(
                    source=source, error=f"missing {TRIPLETS_FILENAME} in archive"
                )
            raw = zf.read(TRIPLETS_FILENAME)
    except zipfile.BadZipFile as exc:
        return TripletExtraction(source=source, error=f"corrupt zip: {exc}")
    except Exception as exc:  # zipfile can raise OSError/RuntimeError on damage
        return TripletExtraction(source=source, error=f"unreadable zip: {exc}")

    return _parse_triplet_payload(source, raw)


def _extract_from_dir(source: PackageSource) -> TripletExtraction:
    pkg_dir = Path(source.path)
    trips_path = pkg_dir / TRIPLETS_FILENAME
    if not trips_path.is_file():
        return TripletExtraction(source=source, error=f"missing {TRIPLETS_FILENAME}")
    try:
        raw = trips_path.read_bytes()
    except OSError as exc:
        return TripletExtraction(source=source, error=f"unreadable triplets.json: {exc}")
    return _parse_triplet_payload(source, raw)


def _parse_triplet_payload(source: PackageSource, raw: bytes) -> TripletExtraction:
    if len(raw) > MAX_TRIPLETS_BYTES:
        return TripletExtraction(
            source=source,
            error=f"triplets.json exceeds {MAX_TRIPLETS_BYTES} bytes — skipped",
        )
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return TripletExtraction(source=source, error=f"invalid JSON: {exc}")

    if not isinstance(data, list):
        return TripletExtraction(source=source, error="triplets.json is not a JSON array")

    valid: List[Dict[str, Any]] = []
    dropped = 0
    for row in data:
        try:
            RerankerTriplet(**row)
            valid.append(row)
        except Exception:
            dropped += 1
    if not valid:
        return TripletExtraction(
            source=source,
            error="no valid triplets (all rows failed RerankerTriplet validation)",
            n_dropped=dropped,
        )
    if dropped:
        logger.warning(
            "Pipeline B: %s dropped %d malformed triplet row(s).",
            source.label, dropped,
        )
    return TripletExtraction(source=source, triplets=valid, n_dropped=dropped)
