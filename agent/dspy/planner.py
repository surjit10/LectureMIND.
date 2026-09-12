# agent/dspy/planner.py
# E1 — DSPy Route Planner.
#
# ONLY classifies the query into a retrieval route.
# DSPy is NOT an agent. NOT answer generation. NOT retrieval.
#
# Allowed routes: graph_only, vector_only, graph+vector
# Uses heuristic keyword matching with optional DSPy ChainOfThought upgrade.
#
# V7 additions (backward compatible):
#   - Richer intent taxonomy (factual QA, conceptual, comparison, etc.)
#   - Summary-aware planning: marks lecture-wide tasks so ContextBuilder
#     can retrieve representative coverage rather than top-K similarity.
#   - plan_full() returns a QueryPlan for downstream components.
#   - plan() continues to return RetrievalRoute (workflow unchanged).

import logging
import re
from typing import Any, Optional

from schemas.enums import RetrievalRoute
from agent.dspy.query_plan import QueryPlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retrieval candidate pool sizes (consumed by vector_retriever_node via
# QueryPlan.top_k).
#
# NORMAL_TOP_K=15: a diagnostic found 19/28 retrieval misses had the correct
# chunk at cosine rank 6-46 with a mean score of 0.652, but the reranker can
# only recover candidates that enter its pool. A larger pool feeds the
# RERANKER; the final LLM context budget (context_budget) and evidence gating are
# unchanged. Lecture-wide retrieval keeps its existing behavior.
# ---------------------------------------------------------------------------
NORMAL_TOP_K = 15
LECTURE_WIDE_TOP_K = 15

# ---------------------------------------------------------------------------
# Keyword patterns — compiled for efficiency
# ---------------------------------------------------------------------------

# Graph routing: relational, structural, dependency questions.
GRAPH_PATTERNS = [
    r"\bprerequisite\b",
    r"\brelated\b",
    r"\brelation\b",
    r"\bconnect(ed|ion)?\b",
    r"\bbefore\b.{0,30}\bafter\b",
    r"\btaught before\b",
    r"\btaught after\b",
    r"\bwhy\b.{0,20}\bbefore\b",
    r"\bdepend(s|ency|encies)?\b",
    r"\brequire[sd]?\b",
    r"\bpath\b",
    r"\blink\b",
    r"\brelationship\b",
    r"\bleads?\s+to\b",
    r"\bbuilt\s+on\b",
    r"\bfound(ation|ational)?\b",
    r"\buse[sd]\s+by\b",
]

# Vector routing: semantic, explanation, content questions.
# Keep specific: avoid question-starters that also appear in graph queries.
VECTOR_PATTERNS = [
    r"\bsummariz(e|ing)\b",
    r"\bexplain\b",
    r"\bwhat\s+was\s+said\b",
    r"\bdescribe\b",
    r"\bsection\b",
    r"\bdetail\b",
    r"\btell\s+me\s+about\b",
    r"\bhow\s+does\b",
    r"\bdefine\b",
    r"\bmeaning\b",
]


# Summary / lecture-wide patterns.  These trigger is_lecture_wide=True.
# "is"/"does" are optional so common informal phrasings match:
#   "what explained in this lecture", "what's explained...",
#   "what is this lecture explaining", "explain what is in this lecture".
SUMMARY_PATTERNS = [
    r"\bsummar(y|ize|ization|ise)\b",
    r"\brecap\b",
    r"\boverview\b",
    r"\blecture\s+summary\b",
    r"\bgive\s+(me\s+)?a\s+summary\b",
    r"\bwhole\s+lecture\b",
    r"\bentire\s+lecture\b",
    r"\ball\s+(of\s+)?the\s+(topics|concepts|content)\b",
    r"\bkeypoints?\b",
    r"\bkey\s+points?\b",
    r"\bmain\s+(points?|ideas?|topics?)\b",
    r"\bexplain\s+(the|this|today'?s?)\s+lecture\b",
    r"\bexplain\w*\s+what\s+(is\s+)?in\s+(this|the)\s+lecture\b",
    r"\bwhat(?:'?s|\s+is)?\s+(?:explain\w*|covered|taught|discussed)\s+in\s+(this|the)\s+lecture\b",
    r"\bwhat(?:'?s|\s+is)?\s+(the|this|today'?s?)\s+lecture\s+(about|on|covering|explain\w*)\b",
    r"\bwhat\s+does\s+(this|the)\s+lecture\s+(cover|explain|teach|discuss|contain)\b",
    r"\blecture\s+(covers|explains|teaches|discusses|is\s+about)\b",
    r"\bkey\s+(concepts|takeaways)\b",
    r"\btopics\s+(are\s+)?covered\b",
    r"\bwhat\s+(topics|concepts|ideas)\s+(are\s+)?(covered|discussed|taught|explained)\b",
    r"\bwhat\s+did\s+we\s+learn\b",
    r"\boverview\s+of\s+(the|this|today'?s?)\s+lecture\b",
    r"\bsyllabus\b",
    r"\bgrading\b",
    r"\bgrading\s+breakdown\b",
    r"\bgrading\s+policy\b",
    r"\bcourse\s+outline\b",
    r"\bcurriculum\b",
    r"\bclass\s+schedule\b",
]

# Learning content patterns — also lecture-wide.
NOTES_PATTERNS = [r"\bnotes?\b", r"\bgenerate\s+notes?\b", r"\bstudy\s+notes?\b"]
FLASHCARD_PATTERNS = [r"\bflashcards?\b", r"\bstudy\s+cards?\b", r"\bkey\s+terms\b"]
QUIZ_PATTERNS = [r"\bquiz\b", r"\btest\s+me\b", r"\bquestions?\b", r"\bpractice\b"]
LEARNING_PATH_PATTERNS = [r"\blearning\s+path\b", r"\bhow\s+to\s+study\b", r"\bwhat\s+to\s+learn\b"]

# Comparison patterns.
COMPARISON_PATTERNS = [
    r"\bcompare\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bdifference\b",
    r"\bsimilar(ity|ities)?\b",
    r"\bbetter\b",
    r"\bworse\b",
]

# Definition patterns.
DEFINITION_PATTERNS = [
    r"\bdefin(e|ition)\b",
    r"\bwhat\s+does\b.{0,20}\bmean\b",
    r"\bmeaning\s+of\b",
    r"\bterm\b",
]

# Reasoning patterns.
REASONING_PATTERNS = [
    r"\bwhy\b",
    r"\bcause\b",
    r"\bcauses?\b",
    r"\bbecause\b",
    r"\bwhat\s+happens\b",
    r"\bwhat\s+if\b",
    r"\bimpact\b",
    r"\beffect\b",
]


def _matches(query_lower: str, patterns: list) -> bool:
    """Return True if ANY pattern matches the query."""
    return any(re.search(p, query_lower) for p in patterns)


def _score(query_lower: str, patterns: list) -> int:
    """Count matching patterns."""
    return sum(1 for p in patterns if re.search(p, query_lower))


class QueryPlanner:
    """
    Classifies queries into retrieval routes using heuristic matching.

    Can optionally be upgraded with a DSPy ChainOfThought module
    via set_dspy_module() for LLM-based classification.

    V7: plan_full() returns a QueryPlan with intent, answer_style,
    lecture-wide flag, and retrieval weight hints.  plan() is unchanged.
    """

    def __init__(self, dspy_module: Optional[Any] = None):
        self._dspy_module = dspy_module

    def set_dspy_module(self, module: Any) -> None:
        """Inject a DSPy classification module."""
        self._dspy_module = module

    def plan(self, query: str) -> RetrievalRoute:
        """
        Classify a query into a retrieval route.

        Args:
            query: User query string.

        Returns:
            One of: RetrievalRoute.graph_only, .vector_only, .graph_and_vector
        """
        return self.plan_full(query).retrieval_route

    def plan_full(self, query: str) -> QueryPlan:
        """
        Return a full QueryPlan for the given query.

        Provides intent, answer_style, lecture-wide flag, and retrieval
        weight hints in addition to the retrieval route.  Used by
        ContextBuilder and learning service; not consumed by workflow.run().
        """
        if not query.strip():
            return QueryPlan(retrieval_route=RetrievalRoute.vector_only)

        if self._dspy_module is not None:
            return self._plan_full_with_dspy(query)

        return self._plan_full_heuristic(query)

    # ------------------------------------------------------------------
    # Internal: heuristic full plan
    # ------------------------------------------------------------------

    def _plan_full_heuristic(self, query: str) -> QueryPlan:
        """Classify via keyword heuristics; return a full QueryPlan."""
        q = query.lower()

        graph_score = _score(q, GRAPH_PATTERNS)
        vector_score = _score(q, VECTOR_PATTERNS)

        # --- Determine retrieval route FIRST (preserves backward compat) ---
        # This matches the original planner logic exactly so existing tests pass.
        if graph_score > 0 and vector_score > 0:
            route = RetrievalRoute.graph_and_vector
        elif graph_score > 0:
            route = RetrievalRoute.graph_only
        else:
            route = RetrievalRoute.vector_only

        # --- Determine intent (does NOT override route) ---
        if _matches(q, LEARNING_PATH_PATTERNS):
            intent = "learning_path"
            is_lecture_wide = True
            answer_style = "structured"
        elif _matches(q, QUIZ_PATTERNS):
            intent = "quiz"
            is_lecture_wide = True
            answer_style = "structured"
        elif _matches(q, FLASHCARD_PATTERNS):
            intent = "flashcards"
            is_lecture_wide = True
            answer_style = "structured"
        elif _matches(q, NOTES_PATTERNS):
            intent = "notes"
            is_lecture_wide = True
            answer_style = "structured"
        elif _matches(q, SUMMARY_PATTERNS):
            intent = "summary"
            is_lecture_wide = True
            answer_style = "detailed"
        elif _matches(q, COMPARISON_PATTERNS):
            intent = "comparison"
            is_lecture_wide = False
            answer_style = "comparative"
        elif _matches(q, DEFINITION_PATTERNS):
            intent = "definition"
            is_lecture_wide = False
            answer_style = "concise"
        elif _matches(q, REASONING_PATTERNS):
            intent = "reasoning"
            is_lecture_wide = False
            answer_style = "detailed"
        elif graph_score > 0 and vector_score > 0:
            intent = "conceptual"
            is_lecture_wide = False
            answer_style = "detailed"
        elif graph_score > 0:
            intent = "factual_qa"
            is_lecture_wide = False
            answer_style = "concise"
        else:
            intent = "factual_qa"
            is_lecture_wide = False
            answer_style = "concise"

        # --- Weight calculation ---
        total = max(graph_score + vector_score, 1)
        graph_weight = round(graph_score / total, 2)
        vector_weight = round(1.0 - graph_weight, 2)

        # --- top_k and context_budget by scope ---
        top_k = LECTURE_WIDE_TOP_K if is_lecture_wide else NORMAL_TOP_K
        context_budget = 6000 if is_lecture_wide else 4000

        # --- need_visual ---
        need_visual = _matches(q, [r"\bdiagram\b", r"\bfigure\b", r"\bimage\b", r"\bslide\b", r"\bvisual\b"])

        logger.debug(
            "Planner: intent=%s, route=%s, lecture_wide=%s, q='%s'",
            intent, route.value, is_lecture_wide, query[:80],
        )

        return QueryPlan(
            retrieval_route=route,
            intent=intent,
            is_lecture_wide=is_lecture_wide,
            answer_style=answer_style,
            graph_weight=graph_weight,
            vector_weight=vector_weight,
            graph_depth=3 if is_lecture_wide else 2,
            top_k=top_k,
            context_budget=context_budget,
            need_visual=need_visual,
        )

    # ------------------------------------------------------------------
    # Internal: heuristic route-only (used by plan() legacy path)
    # ------------------------------------------------------------------

    def _plan_heuristic(self, query: str) -> RetrievalRoute:
        """Classify via keyword heuristics (route only)."""
        return self._plan_full_heuristic(query).retrieval_route

    # ------------------------------------------------------------------
    # Internal: DSPy-based planning
    # ------------------------------------------------------------------

    def _plan_with_dspy(self, query: str) -> RetrievalRoute:
        """Classify via DSPy module."""
        try:
            result = self._dspy_module(query=query)
            route_str = getattr(result, "route", "vector_only").strip().lower()

            route_map = {
                "graph_only": RetrievalRoute.graph_only,
                "vector_only": RetrievalRoute.vector_only,
                "graph+vector": RetrievalRoute.graph_and_vector,
            }
            return route_map.get(route_str, RetrievalRoute.vector_only)
        except Exception as exc:
            logger.warning("DSPy planner failed, falling back to heuristic: %s", exc)
            return self._plan_heuristic(query)

    def _plan_full_with_dspy(self, query: str) -> QueryPlan:
        """Classify via DSPy module, wrap in QueryPlan."""
        try:
            result = self._dspy_module(query=query)
            route_str = getattr(result, "route", "vector_only").strip().lower()

            route_map = {
                "graph_only": RetrievalRoute.graph_only,
                "vector_only": RetrievalRoute.vector_only,
                "graph+vector": RetrievalRoute.graph_and_vector,
            }
            route = route_map.get(route_str, RetrievalRoute.vector_only)
            # Build a minimal plan from DSPy output; heuristic fills the rest.
            heuristic_plan = self._plan_full_heuristic(query)
            return QueryPlan(
                retrieval_route=route,
                intent=heuristic_plan.intent,
                is_lecture_wide=heuristic_plan.is_lecture_wide,
                answer_style=heuristic_plan.answer_style,
                graph_weight=heuristic_plan.graph_weight,
                vector_weight=heuristic_plan.vector_weight,
                graph_depth=heuristic_plan.graph_depth,
                top_k=heuristic_plan.top_k,
                context_budget=heuristic_plan.context_budget,
                need_visual=heuristic_plan.need_visual,
            )
        except Exception as exc:
            logger.warning("DSPy planner failed, falling back to heuristic: %s", exc)
            return self._plan_full_heuristic(query)
