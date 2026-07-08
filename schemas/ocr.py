# schemas/ocr.py
# OCRResult produced by Stage A5 (PaddleOCR 2.9.1).
# A6 joins `ocr_text` list into a single string.

from typing import List
from pydantic import BaseModel, ConfigDict


class OCRResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: int
    ocr_text: List[str]
