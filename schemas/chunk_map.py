# schemas/chunk_map.py
# ChunkSegmentMap — optional performance artifact produced immediately after Stage A7.
#
# Schema:
# {
#   "lec_001_chunk_000001": "seg_5",
#   "lec_001_chunk_000002": "seg_5"
# }
#
# BACKWARD COMPATIBILITY: segments.json remains the source of truth.
# This artifact is a derived convenience lookup only.
# No component may depend EXCLUSIVELY on chunk_segment_map.json.

from typing import Dict
from pydantic import BaseModel, ConfigDict


class ChunkSegmentMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Mapping of chunk_id → segment_id.
    mapping: Dict[str, str]
