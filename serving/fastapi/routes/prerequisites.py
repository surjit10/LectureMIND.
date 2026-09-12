# serving/fastapi/routes/prerequisites.py
# Socratic Prerequisite Back-Tracker: Dedicated API Routes.
#
# Endpoints:
# - GET  /lecture/{lecture_id}/prerequisites/{concept_name}
# - POST /lecture/{lecture_id}/prerequisites/enrich

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from config import LocalSettings, SharedSettings
from retrieval.graph_retriever.neo4j_retriever import get_prerequisites_for_concept
from local.loaders.prerequisite_enricher import enrich_and_load
from local.storage.lecture_registry import PROJECT_ROOT

logger = logging.getLogger(__name__)

router = APIRouter()


class PrerequisiteResponse(BaseModel):
    lecture_id: str
    concept: str
    prerequisites: List[Dict[str, Any]] = []
    count: int = 0


class EnrichResponse(BaseModel):
    lecture_id: str
    status: str
    edges_inferred: int
    loaded_into_neo4j: int
    metrics: Dict[str, Any] = {}


@router.get("/lecture/{lecture_id}/prerequisites/{concept_name}", response_model=PrerequisiteResponse)
async def get_prerequisites_endpoint(
    lecture_id: str,
    concept_name: str,
    min_confidence: float = Query(0.65, ge=0.0, le=1.0),
    max_depth: int = Query(3, ge=1, le=5),
):
    """
    Trace backward prerequisite dependencies for a concept in a lecture.

    Traverses PREREQUISITE_OF edges in Neo4j within the lecture's scope.
    Preserves actual path topology, depth, and intermediate relationship provenance.
    """
    if not lecture_id.strip():
        raise HTTPException(status_code=400, detail="lecture_id must not be empty.")
    if not concept_name.strip():
        raise HTTPException(status_code=400, detail="concept_name must not be empty.")

    # Get Neo4j driver from active workflow if available
    from serving.fastapi.routes.query import get_workflow
    workflow = get_workflow()
    driver = getattr(workflow, "_neo4j_driver", None)

    try:
        items = get_prerequisites_for_concept(
            concept_name=concept_name,
            lecture_id=lecture_id,
            driver=driver,
            min_confidence=min_confidence,
            max_depth=max_depth,
        )
    except Exception as exc:
        logger.error("Error retrieving prerequisites from Neo4j: %s", exc)
        items = []

    # Fallback: if Neo4j returned no items, check prerequisites.json on disk
    if not items:
        pkg_dir = PROJECT_ROOT / "data" / "packages" / lecture_id
        prereqs_path = pkg_dir / "prerequisites.json"
        if prereqs_path.exists():
            try:
                data = json.loads(prereqs_path.read_text(encoding="utf-8"))
                raw_prereqs = data.get("prerequisites", []) if isinstance(data, dict) else data
                # Filter for edges where target matches concept_name
                matched = [
                    p for p in raw_prereqs
                    if p.get("target_name", "").lower() == concept_name.lower()
                    and p.get("confidence", 0.0) >= min_confidence
                ]
                for m in matched:
                    items.append({
                        "concept": m.get("source_name", ""),
                        "entity_id": m.get("source_id", ""),
                        "type": m.get("source_type", "Concept"),
                        "depth": 1,
                        "timestamp": m.get("evidence_timestamp", 0.0),
                        "chunk_id": m.get("evidence_chunk_id", ""),
                        "confidence": m.get("confidence", 0.0),
                        "parent_concept": concept_name,
                        "target_concept": concept_name,
                        "path": [m.get("source_name", ""), concept_name],
                        "edge_provenance": [{
                            "source": m.get("source_name", ""),
                            "target": concept_name,
                            "confidence": m.get("confidence", 0.0),
                            "chunk_id": m.get("evidence_chunk_id", ""),
                            "timestamp": m.get("evidence_timestamp", 0.0),
                        }],
                    })
            except Exception as read_exc:
                logger.warning("Could not read prerequisites.json fallback: %s", read_exc)

    return PrerequisiteResponse(
        lecture_id=lecture_id,
        concept=concept_name,
        prerequisites=items,
        count=len(items),
    )


@router.post("/lecture/{lecture_id}/prerequisites/enrich", response_model=EnrichResponse)
async def enrich_lecture_prerequisites_endpoint(
    lecture_id: str,
    min_confidence: float = Query(0.65, ge=0.0, le=1.0),
):
    """
    Enrich an existing lecture package with inferred prerequisites and load them into Neo4j.
    """
    pkg_dir = PROJECT_ROOT / "data" / "packages" / lecture_id
    if not pkg_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Lecture package directory not found: {pkg_dir}",
        )

    from serving.fastapi.routes.query import get_workflow
    workflow = get_workflow()
    driver = getattr(workflow, "_neo4j_driver", None)

    try:
        result = enrich_and_load(
            package_dir=pkg_dir,
            lecture_id=lecture_id,
            driver=driver,
            min_confidence=min_confidence,
        )
        return EnrichResponse(
            lecture_id=lecture_id,
            status="success",
            edges_inferred=len(result.get("prerequisites", [])),
            loaded_into_neo4j=result.get("loaded_into_neo4j", 0),
            metrics=result.get("metrics", {}),
        )
    except Exception as exc:
        logger.error("Enrichment failed for lecture %s: %s", lecture_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))
