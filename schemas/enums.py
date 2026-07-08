# schemas/enums.py
# All CLOSED enums for LECTUREMIND V2.
# No additional values are permitted without updating the specification first.

from enum import Enum


class EntityType(str, Enum):
    """Closed enum for entity node types in the Neo4j graph."""
    Concept = "Concept"
    Algorithm = "Algorithm"
    Formula = "Formula"
    Code = "Code"
    Diagram = "Diagram"
    LectureSegment = "LectureSegment"


class RelationType(str, Enum):
    """Closed enum for edge relation types in the Neo4j graph."""
    PREREQUISITE_OF = "PREREQUISITE_OF"
    INTRODUCED_BEFORE = "INTRODUCED_BEFORE"
    USED_BY = "USED_BY"
    DERIVED_FROM = "DERIVED_FROM"
    VISUALIZED_BY = "VISUALIZED_BY"
    EXPLAINS = "EXPLAINS"


class RetrievalRoute(str, Enum):
    """Closed enum for DSPy planner routing decisions."""
    graph_only = "graph_only"
    vector_only = "vector_only"
    graph_and_vector = "graph+vector"


class PipelineStatus(str, Enum):
    """Closed enum for the ingestion pipeline state machine lifecycle."""
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    SEGMENTING = "SEGMENTING"
    EXTRACTING_ENTITIES = "EXTRACTING_ENTITIES"
    BUILDING_GRAPH = "BUILDING_GRAPH"
    GENERATING_EMBEDDINGS = "GENERATING_EMBEDDINGS"
    TRAINING_RERANKER = "TRAINING_RERANKER"
    PACKAGING = "PACKAGING"
    READY = "READY"
    FAILED = "FAILED"
