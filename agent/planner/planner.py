# agent/planner/planner.py
# E1 — Heuristic route planner (regex keyword rules).
#
# ONLY classifies the query into a retrieval route.
# The planner only routes and plans: it does not retrieve or generate.
#
# Allowed routes: graph_only, vector_only, graph+vector
# Uses heuristic keyword matching. An optional classifier (e.g. a DSPy module —
# DSPy is not a dependency) can be injected via set_dspy_module().
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
from agent.planner.query_plan import QueryPlan

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

# LLM context character budgets (passed to ContextBuilder via QueryPlan).
# NORMAL was 4000: only ~3 chunks fit, so gold chunks ranked 4-5 (and, on
# graph-routed questions, often rank 2-3) never reached the prompt. 6000 with a
# 25% graph share raised context recall 0.767 -> 0.826 on the 45-question smoke
# set (evaluation/reports/ablation_20261114/), at ~570 extra prompt tokens.
NORMAL_CONTEXT_BUDGET = 6000
LECTURE_WIDE_CONTEXT_BUDGET = 6000

# ---------------------------------------------------------------------------
# Keyword patterns — compiled for efficiency
# ---------------------------------------------------------------------------

# Graph routing: relational, structural, dependency questions.
GRAPH_PATTERNS = [
    # Plurals matter: "What are the prerequisites of X?" is the most common
    # graph question and must not fall through to vector_only.
    r"\bprerequisites?\b",
    r"\brelated\b",
    r"\brelations?\b",
    # "relate"/"relates"/"relating" ("how does paging relate to ...").
    # "relational" (relational model/algebra) is deliberately NOT matched.
    r"\brelat(e|es|ing)\b",
    # connect / connects / connected / connecting / connection(s) — except
    # the neural-network term "fully connected" / "fully-connected".
    r"(?<!fully\s)(?<!fully-)\bconnect(s|ed|ing|ion|ions)?\b",
    r"\bbefore\b.{0,30}\bafter\b",
    r"\btaught before\b",
    r"\btaught after\b",
    r"\bwhy\b.{0,20}\bbefore\b",
    r"\bdepend(s|ency|encies)?\b",
    r"\brequire[sd]?\b",
    r"\bpaths?\b",
    r"\blinks?\b",
    r"\brelationships?\b",
    r"\bleads?\s+to\b",
    r"\bbuilt\s+on\b",
    r"\bfound(ation|ations|ational)?\b",
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
    # NOTE: syllabus / grading / course outline / curriculum / class schedule
    # are NOT lecture-wide: they are specific facts stated in one part of a
    # lecture, so they must go through semantic retrieval. (Routing them to
    # lecture-wide timeline sampling skipped semantic search entirely.)
]

# Learning content patterns — also lecture-wide.
# Kept specific: bare "note" / "question" / "practice" are ordinary words in
# normal questions ("I have a question about paging", "best practice for
# locking") and must not turn a question into a notes/quiz request.
NOTES_PATTERNS = [
    r"\bnotes\b",
    r"\bstudy\s+notes?\b",
    r"\b(generate|make|create|write|take|prepare|give\s+me)\b.{0,20}\bnote\b",
]
FLASHCARD_PATTERNS = [r"\bflashcards?\b", r"\bstudy\s+cards?\b", r"\bkey\s+terms\b"]
QUIZ_PATTERNS = [
    r"\bquiz(zes)?\b",
    r"\btest\s+me\b",
    r"\bpractice\s+(questions?|problems?|exercises?|tests?|exams?)\b",
    r"\b(generate|create|make|write|prepare|give\s+me)\b.{0,30}\b(questions|mcqs?)\b",
    r"\b(exam|mcq|multiple[\s-]choice)\s+questions?\b",
]
LEARNING_PATH_PATTERNS = [r"\blearning\s+path\b", r"\bhow\s+to\s+study\b", r"\bwhat\s+to\s+learn\b"]

# Words that only describe the *kind of request* (summarise, overview, notes,
# quiz, ...) or the lecture as a whole. A summary/notes/quiz-style query made
# only of these words is lecture-wide; any other word names a topic ("give an
# overview of virtual memory", "quiz me on paging"), and such a query must use
# semantic retrieval instead of timeline sampling (which ignores the query).
_SCOPE_GENERIC_WORDS = {
    # request verbs / politeness
    "give", "gave", "provide", "write", "make", "create", "generate", "prepare",
    "list", "tell", "show", "please", "can", "could", "would", "will", "you",
    "explain", "explained", "explaining", "explains", "describe", "go", "over",
    "through", "walk", "help", "want", "need", "like",
    # summary / overview vocabulary
    "summary", "summaries", "summarize", "summarise", "summarizing", "summarising",
    "summarization", "summarisation", "recap", "overview", "outline", "brief",
    "briefly", "short", "quick", "quickly", "detailed", "key", "main", "important",
    "points", "point", "ideas", "idea", "topics", "topic", "concepts", "concept",
    "takeaways", "takeaway", "highlights", "highlight", "keypoints", "keypoint",
    "content", "contents", "everything", "things", "stuff",
    # scope words (the lecture as a whole)
    "lecture", "lectures", "class", "course", "session", "lesson", "video",
    "talk", "today", "todays", "whole", "entire", "all", "each", "every",
    "section", "sections", "part", "parts", "slide", "slides", "far",
    # coverage verbs
    "cover", "covers", "covered", "covering", "discuss", "discusses",
    "discussed", "teach", "teaches", "taught", "learn", "learned", "learnt",
    "learning", "contain", "contains", "said", "mentioned", "presented",
    # learning-content vocabulary
    "notes", "note", "study", "flashcards", "flashcard", "cards", "card",
    "terms", "term", "quiz", "quizzes", "test", "questions", "question",
    "practice", "exam", "mcq", "mcqs", "path", "how", "what", "to",
    # function words
    "the", "a", "an", "this", "that", "these", "those", "is", "are", "was",
    "were", "be", "been", "do", "does", "did", "done", "have", "has", "had",
    "of", "on", "in", "at", "by", "for", "from", "about", "with", "and", "or",
    "it", "its", "we", "us", "me", "my", "our", "i", "so", "up", "some", "any",
    "which", "there", "here", "now", "then", "s",
}


def _has_topic_terms(query_lower: str) -> bool:
    """True when the query names something beyond generic request/scope words.

    Pure numbers ("summarize lecture 3") are scope, not topics.
    """
    return any(
        tok not in _SCOPE_GENERIC_WORDS and not tok.isdigit()
        for tok in re.findall(r"[a-z0-9]+", query_lower)
    )

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

        # A summary/notes/quiz-style request that names a specific topic
        # ("overview of virtual memory", "quiz me on paging") is scoped to
        # that topic: keep the intent and answer style, but use semantic
        # retrieval. Lecture-wide timeline sampling ignores the query text.
        if is_lecture_wide and _has_topic_terms(q):
            is_lecture_wide = False

        # --- Weight calculation ---
        total = max(graph_score + vector_score, 1)
        graph_weight = round(graph_score / total, 2)
        vector_weight = round(1.0 - graph_weight, 2)

        # --- top_k and context_budget by scope ---
        top_k = LECTURE_WIDE_TOP_K if is_lecture_wide else NORMAL_TOP_K
        context_budget = LECTURE_WIDE_CONTEXT_BUDGET if is_lecture_wide else NORMAL_CONTEXT_BUDGET

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
