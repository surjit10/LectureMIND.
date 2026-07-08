# evaluation/metrics/answer_metrics.py
import logging
from typing import List
import re

logger = logging.getLogger(__name__)

def calculate_answer_similarity(ground_truth: str, generated_answer: str) -> float:
    """
    Placeholder for advanced similarity (e.g. BERTScore, RAGAS).
    For now, computes simple Jaccard similarity of words to keep it modular.
    """
    if not ground_truth and not generated_answer:
        return 1.0
    if not ground_truth or not generated_answer:
        return 0.0
        
    set1 = set(re.findall(r'\w+', ground_truth.lower()))
    set2 = set(re.findall(r'\w+', generated_answer.lower()))
    
    if not set1 and not set2:
        return 1.0
        
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    return len(intersection) / len(union)

def calculate_keyword_recall(expected_keywords: List[str], generated_answer: str) -> float:
    """Checks how many expected keywords are present in the answer."""
    if not expected_keywords:
        return 1.0
    if not generated_answer:
        return 0.0
        
    text = generated_answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text)
    return hits / len(expected_keywords)

def get_context_length(final_context: str) -> int:
    """Returns the character length of the final context string."""
    return len(final_context) if final_context else 0

def get_answer_length(generated_answer: str) -> int:
    """Returns the character length of the generated answer string."""
    return len(generated_answer) if generated_answer else 0
