# agent/langgraph/nodes/answer_generator.py
# E4 — Answer Generator.
#
# Generates a grounded answer using final_context.
# If evidence is insufficient, returns a disclaimer.
# Does NOT hallucinate. Does NOT use model prior knowledge.
#
# Delegates LLM calls to the active LLMBackend (Ollama or any online provider).
# Populates: answer, sources, graph_path.

import logging
from typing import Any, Dict, List, Optional, Tuple

from config import LocalSettings
from local.llm.base import LLMBackend

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an intelligent and helpful lecture assistant. "
    "Answer the student's question using ONLY the evidence in the provided context. "
    "Priority order: Transcript content > OCR slide text > Visual descriptions > Graph relationships. "
    "Graph lines look like '[Graph] EntityA -RELATION-> EntityB' and describe "
    "relationships between lecture concepts; use them to answer relationship "
    "questions (e.g. 'how does X relate to Y', 'what depends on Z') and to "
    "connect entities when the transcript is sparse.\n"
    "Rules you MUST follow:\n"
    "  1. Ground every sentence in the retrieved context. Do not use prior knowledge. Never hallucinate, speculate, or invent facts.\n"
    "  2. Answer directly when evidence exists. Merge related information from multiple chunks into a cohesive explanation.\n"
    "  3. Distinguish factual QA from summary-style questions (e.g., 'key takeaways', 'summary', 'main ideas'). For the latter, produce natural summaries by synthesizing the retrieved facts. You do not need explicit keywords in the text to deduce main ideas.\n"
    "  4. Synthesize across sources instead of narrating slide-by-slide.\n"
    "  5. ONLY if the retrieved context is genuinely unrelated or too sparse to form any relevant answer, respond with exactly: 'Insufficient evidence found in lecture.' If partial or synthesized evidence exists, use it.\n"
    "  6. Respond in the same language as the student's question. If the question is in English, answer entirely in English — never mix words or characters from other languages into an English answer."
)


def _format_timestamp(ts: float) -> str:
    """Convert seconds to HH:MM:SS."""
    hours = int(ts // 3600)
    minutes = int((ts % 3600) // 60)
    seconds = int(ts % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _build_sources(reranked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build sources from reranked_results.

    Sources originate ONLY from reranked_results.
    Never invents chunk_ids, timestamps, or segment_ids.
    """
    sources = []
    seen = set()
    for result in reranked_results:
        payload = result.get("payload", result)
        chunk_id = payload.get("chunk_id", result.get("chunk_id", ""))
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)

        timestamp = payload.get("timestamp", 0)
        source = {
            "chunk_id": chunk_id,
            "timestamp": _format_timestamp(float(timestamp)),
            "segment_id": payload.get("segment_id", ""),
        }
        sources.append(source)

    return sources


def _build_graph_path(graph_results: List[Dict[str, Any]]) -> List[str]:
    """
    Build graph_path from graph retrieval results.

    Only populated if graph retrieval was used.
    """
    path = []
    seen = set()
    for result in graph_results:
        for key in ["start_name", "related_name"]:
            name = result.get(key)
            if name and name not in seen:
                seen.add(name)
                path.append(name)
    return path


def answer_generator_node(
    state: Dict[str, Any],
    llm_backend: Optional[LLMBackend] = None,
    ollama_client: Optional[Any] = None,  # kept for backward compat with existing tests
    local_settings: LocalSettings | None = None,
) -> Dict[str, Any]:
    """
    LangGraph node: generate grounded answer.

    Populates ONLY: answer, sources, graph_path.

    Args:
        state:          Current pipeline state.
        llm_backend:    Active LLMBackend (preferred). If None, falls back to
                        OllamaBackend using local_settings / ollama_client.
        ollama_client:  Deprecated — kept for backward compat in tests.
        local_settings: Injected LocalSettings (used for Ollama fallback).
    """
    # Resolve backend.
    backend = _resolve_backend(llm_backend, ollama_client, local_settings)

    settings = local_settings or LocalSettings()
    query = state.get("query", "")
    final_context = state.get("final_context", "")
    reranked_results = state.get("reranked_results", [])
    graph_results = state.get("graph_results", [])

    # Derive answer_style from the query plan.
    try:
        from agent.dspy.planner import QueryPlanner
        plan = QueryPlanner().plan_full(query)
        answer_style = plan.answer_style
    except Exception:
        answer_style = "concise"

    # Build sources from reranked results (never invented).
    sources = _build_sources(reranked_results)

    # Build graph path (only if graph was used).
    graph_path = _build_graph_path(graph_results)

    # Generate answer.
    metadata = {}
    if not final_context.strip():
        answer = "Insufficient evidence found in lecture."
    else:
        answer, metadata = _generate_answer(
            query, final_context, backend, answer_style=answer_style
        )

    return {
        "answer": answer,
        "sources": sources,
        "graph_path": graph_path,
        "llm_metadata": metadata,
    }



def _resolve_backend(
    llm_backend: Optional[LLMBackend],
    ollama_client: Optional[Any],
    local_settings: Optional[LocalSettings],
) -> LLMBackend:
    """Resolve the LLMBackend to use, with a layered fallback."""
    if llm_backend is not None:
        return llm_backend
    # Backward-compat path: wrap injected ollama_client.
    if ollama_client is not None:
        from local.llm.ollama_backend import OllamaBackend
        model = (local_settings or LocalSettings()).OLLAMA_MODEL
        return OllamaBackend(model=model, ollama_client=ollama_client)
    # Default: use the active backend from provider registry.
    try:
        from local.llm.provider_registry import get_provider_registry
        return get_provider_registry().get_active_backend()
    except Exception:
        from local.llm.ollama_backend import OllamaBackend
        model = (local_settings or LocalSettings()).OLLAMA_MODEL
        return OllamaBackend(model=model)


def _generate_answer(
    query: str,
    context: str,
    backend: LLMBackend,
    answer_style: str = "concise",
) -> Tuple[str, dict]:
    """Generate a grounded answer using the active LLM backend."""

    _style_instructions = {
        "concise": (
            "Give a brief, direct answer (2–4 sentences). "
            "State only what the context supports."
        ),
        "detailed": (
            "Give a thorough, multi-paragraph answer. "
            "Cover all relevant aspects present in the context. "
            "Synthesize — do not repeat the same point."
        ),
        "structured": (
            "Organize your answer with clear headings and bullet points. "
            "Cover all key topics present in the context."
        ),
        "comparative": (
            "Present a structured comparison. "
            "Address similarities, differences, and trade-offs "
            "as evidenced by the context."
        ),
    }
    style_hint = _style_instructions.get(answer_style, _style_instructions["concise"])

    user_prompt = (
        f"Retrieved lecture context:\n"
        f"{context}\n\n"
        f"Student question: {query}\n\n"
        f"{style_hint} "
        f"Every claim must come from the context above. "
        f"If the context does not support the question, say so explicitly."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result = backend.generate_with_metadata(messages)
    if not result.answer:
        logger.error("E4: LLM backend returned empty response.")
        return "Insufficient evidence found in lecture.", result.metadata
    return result.answer, result.metadata
