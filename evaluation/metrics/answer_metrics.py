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
    """Checks how many expected keywords are present in the answer.

    Returns 0.0 (not 1.0) when no keywords are supplied — an empty keyword
    list must not be reported as perfect recall.
    """
    if not expected_keywords:
        return 0.0
    if not generated_answer:
        return 0.0
        
    text = generated_answer.lower()
    hits = sum(1 for kw in expected_keywords if kw and kw.lower() in text)
    return hits / len(expected_keywords)


def calculate_answer_f1(ground_truth: str, generated_answer: str) -> float:
    """
    Token-level F1 between ground truth and generated answer (SQuAD-style).

    More informative than raw Jaccard: rewards partial overlap while
    penalizing length mismatch.  Returns 0.0 if either side is empty.
    """
    if not ground_truth or not generated_answer:
        return 0.0

    def _tokens(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    gt_tokens = _tokens(ground_truth)
    ans_tokens = _tokens(generated_answer)
    if not gt_tokens or not ans_tokens:
        return 0.0

    from collections import Counter
    gt_counts = Counter(gt_tokens)
    ans_counts = Counter(ans_tokens)

    overlap = sum((gt_counts & ans_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(ans_tokens)
    recall = overlap / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def get_context_length(final_context: str) -> int:
    """Returns the character length of the final context string."""
    return len(final_context) if final_context else 0

def get_answer_length(generated_answer: str) -> int:
    """Returns the character length of the generated answer string."""
    return len(generated_answer) if generated_answer else 0
