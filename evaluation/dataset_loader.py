# evaluation/dataset_loader.py
import json
import logging
from typing import Any, Dict, List
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkSample:
    lecture_id: str
    query: str
    ground_truth_answer: str
    expected_chunk_ids: List[str]
    expected_route: str
    question_type: str
    difficulty: str
    topic: str

class DatasetLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def load(self) -> List[BenchmarkSample]:
        """Loads and validates the benchmark dataset."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        samples = []
        for item in data:
            self._validate_sample(item)
            samples.append(BenchmarkSample(**item))
            
        logger.info(f"Loaded {len(samples)} benchmark samples from {self.dataset_path}")
        return samples

    def _validate_sample(self, item: Dict[str, Any]) -> None:
        """Validates all required fields are present in the sample."""
        required_fields = {
            "lecture_id", "query", "ground_truth_answer",
            "expected_chunk_ids", "expected_route", "question_type",
            "difficulty", "topic"
        }
        missing = required_fields - item.keys()
        if missing:
            raise ValueError(f"Sample missing required fields: {missing}. Sample: {item.get('query', 'Unknown')}")
