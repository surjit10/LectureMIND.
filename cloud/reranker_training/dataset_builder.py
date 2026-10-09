# cloud/reranker_training/dataset_builder.py
#
# Pipeline B, Step 2 — merge every lecture package's triplets.json into ONE
# global training dataset.
#
#   discover → extract (package_discovery) → merge → dedupe → filter → split
#
# Reliability contract:
#   * missing lecture packages   → skipped (logged), job continues
#   * corrupted triplets         → package skipped (logged), job continues
#   * duplicate triplets         → deduplicated by content hash
#   * tiny/noisy lecture sets    → optional per-lecture minimum + cap
#
# Provenance caveat: if packages_root contains BOTH a zip and a separately
# extracted twin dir with the same lecture key, both sources are merged and
# the data is rescued by dedupe; by_lecture then reports the last-seen count
# for that key (stats slightly undercount sources).
#
# Environment: Kaggle GPU notebook (Pipeline B).

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cloud.reranker_training.package_discovery import (
    PackageSource,
    TripletExtraction,
    discover_packages,
    extract_triplets,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetReport:
    """Statistics for one merged dataset build."""

    sources_found: int = 0
    sources_used: int = 0
    sources_skipped: List[Dict[str, str]] = field(default_factory=list)
    n_triplets_raw: int = 0
    n_triplets_after_dedupe: int = 0
    by_lecture: Dict[str, int] = field(default_factory=dict)
    triplets: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def lecture_ids(self) -> List[str]:
        return sorted(self.by_lecture.keys())


def _triplet_hash(t: Dict[str, Any]) -> str:
    """Content hash used for duplicate detection (ignores key order)."""
    canonical = json.dumps(t, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def build_dataset(
    packages_root: Path,
    exclude_dirs: Optional[List[str]] = None,
    min_triplets: int = 0,
    max_per_lecture: Optional[int] = None,
    dedupe: bool = True,
) -> DatasetReport:
    """
    Discover every package, extract only triplets.json, merge + dedupe.

    Args:
        packages_root: Root directory scanned for knowledge packages.
        exclude_dirs: Directory names skipped during discovery.
        min_triplets: Skip a package when it yields fewer valid triplets than
            this (tiny sets add noise). 0 includes everything.
        max_per_lecture: Optional cap on triplets contributed per package, so a
            single lecture cannot dominate the global dataset.
        dedupe: Remove exact duplicate (query, positive, negative) rows.

    Returns:
        DatasetReport with merged triplets and full provenance.
    """
    sources = discover_packages(Path(packages_root), exclude_dirs=exclude_dirs)

    report = DatasetReport(sources_found=len(sources))
    merged: List[Dict[str, Any]] = []

    for source in sources:
        extraction: TripletExtraction = extract_triplets(source)
        if not extraction.ok:
            report.sources_skipped.append(
                {"source": source.label, "reason": extraction.error or "unknown"}
            )
            logger.warning(
                "Pipeline B: skip package %s — %s",
                source.label, extraction.error,
            )
            continue

        valid = list(extraction.triplets)
        if len(valid) < min_triplets:
            reason = f"only {len(valid)} valid triplets (< min {min_triplets})"
            report.sources_skipped.append({"source": source.label, "reason": reason})
            logger.info("Pipeline B: skip package %s — %s", source.label, reason)
            continue
        if max_per_lecture is not None:
            valid = valid[:max_per_lecture]

        report.sources_used += 1
        report.by_lecture[source.lecture_key] = len(valid)
        merged.extend(valid)

    report.n_triplets_raw = len(merged)

    if dedupe:
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for t in merged:
            h = _triplet_hash(t)
            if h not in seen:
                seen.add(h)
                unique.append(t)
        n_dupes = len(merged) - len(unique)
        if n_dupes:
            logger.info("Pipeline B: removed %d duplicate triplet(s).", n_dupes)
        merged = unique

    report.n_triplets_after_dedupe = len(merged)
    report.triplets = merged
    logger.info(
        "Pipeline B: dataset built — %d package(s) used, %d skipped, "
        "%d unique triplets.",
        report.sources_used, len(report.sources_skipped), len(merged),
    )
    return report


def train_dev_split(
    triplets: List[Dict[str, Any]],
    dev_fraction: float = 0.1,
    seed: int = 42,
    min_dev: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deterministically split merged triplets into train/dev sets by positive
    passage group.

    Guarantees at least `min_dev` dev triplets whenever there are enough rows
    (>= 2). With a single triplet the dev set is empty and post-training
    evaluation is skipped rather than evaluated on training data.  Keeping all
    rows with the same positive passage in one split prevents passage leakage
    when several generated questions target the same lecture chunk.
    """
    n = len(triplets)
    if n == 0:
        return [], []
    if n == 1:
        return list(triplets), []

    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for triplet in triplets:
        # Case/whitespace normalization is enough to group generated variants
        # without altering the text passed to the trainer.
        key = " ".join(str(triplet.get("positive", "")).lower().split())
        groups.setdefault(key, []).append(triplet)

    if len(groups) == 1:
        logger.warning(
            "Pipeline B: all %d triplets share one positive passage; a "
            "leakage-free dev split is impossible, so dev is empty.", n,
        )
        return list(triplets), []

    target_dev = max(min_dev, int(round(n * dev_fraction)))
    target_dev = min(target_dev, n - 1)
    group_items = list(groups.items())
    rng.shuffle(group_items)

    dev: List[Dict[str, Any]] = []
    train: List[Dict[str, Any]] = []
    for _, rows in group_items:
        # Add whole groups until the requested dev size is met.  The result may
        # be slightly larger than target_dev, which is preferable to leakage.
        if len(dev) < target_dev and len(rows) < n:
            dev.extend(rows)
        else:
            train.extend(rows)

    if not train:
        # Move the final complete group back to train.
        final_key, final_rows = group_items[-1]
        train.extend(final_rows)
        final_ids = {id(row) for row in final_rows}
        dev = [row for row in dev if id(row) not in final_ids]
    return train, dev
