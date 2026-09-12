# evaluation/metrics/answer_metrics.py
import logging
from typing import List
import re

logger = logging.getLogger(__name__)

_COMMON_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "again", "further", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "too", "very", "can", "will", "should", "now",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "would", "could",
}


def _stem_simple(token: str) -> str:
    """Lightweight suffix normalization for fair content comparison."""
    t = token.lower()
    if len(t) > 3:
        if t.endswith("ies"):
            return t[:-3] + "y"
        if t.endswith("tions"):
            return t[:-5]
        if t.endswith("tion"):
            return t[:-4]
        if t.endswith("ing"):
            t = t[:-3]
        elif t.endswith("ed"):
            t = t[:-2]
        elif t.endswith("es"):
            t = t[:-2]
            if not t.endswith("e") and not t.endswith(("s", "x", "z", "ch", "sh")):
                t = t + "e"
        elif t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
    return t


def calculate_answer_similarity(ground_truth: str, generated_answer: str, use_content_words: bool = True) -> float:
    """
    Computes semantic-aware content word similarity between ground truth and answer.
    Filters common stopwords and normalizes suffixes to fairly evaluate paraphrased answers.
    """
    if not ground_truth and not generated_answer:
        return 1.0
    if not ground_truth or not generated_answer:
        return 0.0

    tokens1 = [t.lower() for t in re.findall(r'\w+', ground_truth) if len(t) >= 2]
    tokens2 = [t.lower() for t in re.findall(r'\w+', generated_answer) if len(t) >= 2]

    if use_content_words:
        set1 = {_stem_simple(t) for t in tokens1 if t not in _COMMON_STOPWORDS}
        set2 = {_stem_simple(t) for t in tokens2 if t not in _COMMON_STOPWORDS}
    else:
        set1 = set(tokens1)
        set2 = set(tokens2)

    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    return len(intersection) / len(union)


def calculate_keyword_recall(expected_keywords: List[str], generated_answer: str) -> float:
    """Checks how many expected keywords are present in the answer.

    Supports exact match, singular/plural forms, and multi-word token overlap.
    Returns 0.0 when no keywords are supplied.
    """
    if not expected_keywords:
        return 0.0
    if not generated_answer:
        return 0.0

    text = generated_answer.lower()
    text_tokens = set(re.findall(r'\w+', text))

    hits = 0
    for kw in expected_keywords:
        if not kw:
            continue
        kw_lower = kw.lower()
        if kw_lower in text:
            hits += 1
            continue
        kw_tokens = [t for t in re.findall(r'\w+', kw_lower) if t not in _COMMON_STOPWORDS]
        if kw_tokens and all((t in text_tokens or _stem_simple(t) in text) for t in kw_tokens):
            hits += 1
            continue
        if kw_lower.endswith('s') and kw_lower[:-1] in text:
            hits += 1
        elif (kw_lower + 's') in text:
            hits += 1

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
