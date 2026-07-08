# serving/fastapi/learning_service.py
# V7 Part 2 — Structured Learning Content Generation.
#
# Generates Notes, Flashcards, Quiz, and Learning Path from retrieved
# lecture context.  Uses the active QueryWorkflow for retrieval (so the
# same retrieval infrastructure is reused without modification), then
# sends the context through structured LLM prompts via the active
# LLMBackend (Ollama or any configured online provider).
#
# Rules:
#   - Never changes retrieval, reranking, or embedding logic.
#   - Derives all content exclusively from retrieved context.
#   - Explicitly states "Insufficient material" when context is thin.
#   - Priority order: Transcript > OCR > Visuals (preserved by rerank_service).

import json
import logging
import re
from typing import Any, Dict, List, Optional

from local.llm.base import LLMBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared LLM caller
# ---------------------------------------------------------------------------

def _get_backend(llm_backend: Optional[LLMBackend]) -> LLMBackend:
    """Resolve the LLMBackend to use. Falls back to provider registry."""
    if llm_backend is not None:
        return llm_backend
    try:
        from local.llm.provider_registry import get_provider_registry
        return get_provider_registry().get_active_backend()
    except Exception:
        from local.llm.ollama_backend import OllamaBackend
        from config import LocalSettings
        return OllamaBackend(model=LocalSettings().OLLAMA_MODEL)


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    backend: LLMBackend,
) -> str:
    """
    Call the active LLM backend and return the text response.

    Never raises — returns an empty string on failure.
    """
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return backend.generate(messages)
    except Exception as exc:
        logger.error("Learning service: LLM call failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Context builder — reuses existing workflow retrieval
# ---------------------------------------------------------------------------

def _retrieve_context(
    workflow: Any,
    broad_query: str,
    lecture_id: str,
    is_lecture_wide: bool = False,
    need_visual: bool = False,
) -> str:
    """
    Run the retrieval pipeline via the existing QueryWorkflow and return
    the assembled final_context.

    Args:
        workflow:         Active QueryWorkflow (carries Qdrant client + model).
        broad_query:      Query string for semantic retrieval.
        is_lecture_wide:  When True, ContextBuilder samples representative
                          chunks across the lecture timeline instead of
                          taking only the top semantically-similar matches.
        need_visual:      When True, [Visual] labels are included in context.
    """
    from schemas.enums import RetrievalRoute
    from retrieval.reranker.rerank_service import rerank
    from retrieval.vector_retriever.qdrant_retriever import retrieve_vectors

    # Retrieve more chunks for lecture-wide tasks so the timeline
    # sampler has enough input to cover the full lecture span.
    top_k = 20 if is_lecture_wide else 5

    vector_results: List[Dict[str, Any]] = []
    try:
        if is_lecture_wide:
            from retrieval.vector_retriever.qdrant_retriever import retrieve_lecture_wide
            vector_results = retrieve_lecture_wide(
                qdrant_client=workflow._qdrant_client,
                top_k=top_k,
                lecture_id=lecture_id,
            )
        else:
            vector_results = retrieve_vectors(
                broad_query,
                qdrant_client=workflow._qdrant_client,
                embedding_model=workflow._embedding_model,
                top_k=top_k,
                lecture_id=lecture_id,
            )
    except Exception as exc:
        logger.warning("Learning service: Vector retrieval failed: %s", exc)

    try:
        _, final_context = rerank(
            broad_query,
            graph_results=[],
            vector_results=vector_results,
            is_lecture_wide=is_lecture_wide,
            need_visual=need_visual,
        )
    except Exception as exc:
        logger.warning("Learning service: Rerank failed: %s", exc)
        final_context = ""

    return final_context


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

_NOTES_SYSTEM = (
    "You are an expert academic note-taker. "
    "Generate structured Markdown notes using ONLY the provided lecture context. "
    "Do not use any external knowledge or invent information not present in the context. "
    "If the context is insufficient, state: 'Insufficient material retrieved for this lecture.'"
)

_NOTES_USER_TEMPLATE = """\
Lecture context:
{context}

Generate comprehensive, structured Markdown notes with the following sections:
# [Lecture Title — infer from context]

## Overview
[2–3 sentence summary]

## Key Concepts
[Bulleted list of the central ideas]

## Definitions
[Term: concise definition — only for terms explicitly in the context]

## Examples
[Concrete examples from the context]

## Relationships
[How concepts connect to each other, based only on the context]

## Key Takeaways
[3–5 actionable bullet points a student should remember]

Rules:
- No duplicate content across sections.
- Every claim must come from the context above.
- Do not narrate slide-by-slide. Synthesize.
"""


def generate_notes(
    workflow: Any,
    lecture_id: str,
    settings: Optional[Any] = None,
    llm_backend: Optional[LLMBackend] = None,
) -> str:
    backend = _get_backend(llm_backend)
    context = _retrieve_context(
        workflow,
        "main concepts key ideas overview of this lecture",
        lecture_id=lecture_id,
        is_lecture_wide=True,
    )
    if not context.strip():
        return "Insufficient material retrieved for this lecture."
    user = _NOTES_USER_TEMPLATE.format(context=context)
    result = _call_llm(_NOTES_SYSTEM, user, backend)
    return result or "Insufficient material retrieved for this lecture."


# ---------------------------------------------------------------------------
# Flashcards
# ---------------------------------------------------------------------------

_FLASHCARDS_SYSTEM = (
    "You are an expert flashcard creator. "
    "Generate conceptual flashcards using ONLY the provided lecture context. "
    "Focus on: definitions, comparisons, relationships, and applications. "
    "Avoid trivia, dates, and factual minutiae. "
    "Never invent information absent from the context. "
    "Return ONLY a JSON array, no markdown wrapper, no explanation."
)

_FLASHCARDS_USER_TEMPLATE = """\
Lecture context:
{context}

Generate exactly {count} distinct flashcards.
Each flashcard must be a JSON object with keys "question" and "answer".
The question must be conceptual (definition, comparison, or application).
The answer must be concise (1–2 sentences max) and grounded in the context.

Return ONLY a JSON array like:
[
  {{"question": "...", "answer": "..."}},
  ...
]
"""


def _parse_json_array(raw: str) -> List[Dict[str, str]]:
    """Extract a JSON array from LLM output (may be wrapped in markdown)."""
    # Strip markdown code fences.
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
    # Find the first '['.
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("Learning service: JSON parse failed on: %s", raw[:200])
        return []


def generate_flashcards(
    workflow: Any,
    lecture_id: str,
    count: int = 10,
    settings: Optional[Any] = None,
    llm_backend: Optional[LLMBackend] = None,
) -> List[Dict[str, str]]:
    backend = _get_backend(llm_backend)
    context = _retrieve_context(
        workflow,
        "definitions concepts relationships comparisons key terms",
        lecture_id=lecture_id,
        is_lecture_wide=True,
    )
    if not context.strip():
        return []
    user = _FLASHCARDS_USER_TEMPLATE.format(context=context, count=count)
    raw = _call_llm(_FLASHCARDS_SYSTEM, user, backend)
    cards = _parse_json_array(raw)
    # Normalise to only question/answer keys; filter malformed entries.
    result = []
    seen_questions: set = set()
    for card in cards:
        q = str(card.get("question", "")).strip()
        a = str(card.get("answer", "")).strip()
        if q and a and q not in seen_questions:
            seen_questions.add(q)
            result.append({"question": q, "answer": a})
    return result[:count]


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------

_QUIZ_SYSTEM = (
    "You are an expert quiz author. "
    "Generate multiple-choice questions using ONLY the provided lecture context. "
    "Each question must test understanding, not memorisation. "
    "Provide one clearly correct answer and three realistic but incorrect distractors. "
    "Add a brief evidence-based explanation for the correct answer. "
    "Never invent facts not present in the context. "
    "Return ONLY a JSON array, no markdown wrapper, no explanation."
)

_QUIZ_USER_TEMPLATE = """\
Lecture context:
{context}

Generate exactly {count} multiple-choice questions.
Each must be a JSON object with keys:
  "question"    : the question text
  "options"     : list of exactly 4 answer strings
  "correct"     : the exact string from options that is correct
  "explanation" : 1–2 sentence evidence-based explanation

Return ONLY a JSON array like:
[
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "correct": "A",
    "explanation": "..."
  }},
  ...
]
"""


def generate_quiz(
    workflow: Any,
    lecture_id: str,
    count: int = 5,
    settings: Optional[Any] = None,
    llm_backend: Optional[LLMBackend] = None,
) -> List[Dict[str, Any]]:
    backend = _get_backend(llm_backend)
    context = _retrieve_context(
        workflow,
        "key concepts mechanisms processes cause effect applications",
        lecture_id=lecture_id,
        is_lecture_wide=True,
    )
    if not context.strip():
        return []
    user = _QUIZ_USER_TEMPLATE.format(context=context, count=count)
    raw = _call_llm(_QUIZ_SYSTEM, user, backend)
    questions = _parse_json_array(raw)
    result = []
    seen_qs: set = set()
    for q in questions:
        text = str(q.get("question", "")).strip()
        options = q.get("options", [])
        correct = str(q.get("correct", "")).strip()
        explanation = str(q.get("explanation", "")).strip()
        if (
            text
            and isinstance(options, list)
            and len(options) == 4
            and correct in options
            and text not in seen_qs
        ):
            seen_qs.add(text)
            result.append(
                {
                    "question": text,
                    "options": [str(o) for o in options],
                    "correct": correct,
                    "explanation": explanation,
                }
            )
    return result[:count]


# ---------------------------------------------------------------------------
# Learning Path
# ---------------------------------------------------------------------------

_LP_SYSTEM = (
    "You are an expert curriculum designer. "
    "Generate a logical learning path using ONLY the provided lecture context. "
    "Order topics so that prerequisites come first. "
    "Do not invent topics or prerequisites not grounded in the context. "
    "Return ONLY a JSON array, no markdown wrapper, no explanation."
)

_LP_USER_TEMPLATE = """\
Lecture context:
{context}

Based solely on the above, generate a structured learning path.
Each step must be a JSON object with:
  "title"       : short topic name (max 8 words)
  "description" : 1–2 sentences explaining what to learn and why at this step

Order from foundational to advanced as evidenced by the context.
Return ONLY a JSON array.
"""


def generate_learning_path(
    workflow: Any,
    lecture_id: str,
    settings: Optional[Any] = None,
    llm_backend: Optional[LLMBackend] = None,
) -> List[Dict[str, str]]:
    backend = _get_backend(llm_backend)
    context = _retrieve_context(
        workflow,
        "topics prerequisites foundations advanced concepts structure",
        lecture_id=lecture_id,
        is_lecture_wide=True,
    )
    if not context.strip():
        return []
    user = _LP_USER_TEMPLATE.format(context=context)
    raw = _call_llm(_LP_SYSTEM, user, backend)
    steps = _parse_json_array(raw)
    result = []
    for step in steps:
        title = str(step.get("title", "")).strip()
        desc = str(step.get("description", "")).strip()
        if title and desc:
            result.append({"title": title, "description": desc})
    return result
