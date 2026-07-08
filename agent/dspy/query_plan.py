# agent/dspy/query_plan.py
# V7 — Query Plan dataclass.
#
# Carries the full planner output beyond just the retrieval route.
# The workflow's plan() method still returns RetrievalRoute for
# backward compatibility.  plan_full() returns a QueryPlan for
# components that need richer signals (ContextBuilder, learning service).
#
# No LangGraph nodes are changed.  No workflow modifications required.

from dataclasses import dataclass, field
from typing import Literal

from schemas.enums import RetrievalRoute

# Intent taxonomy — maps onto downstream retrieval and generation strategy.
QueryIntent = Literal[
    "factual_qa",       # What is X? What does X mean?
    "conceptual",       # Explain X. How does X work?
    "comparison",       # Compare X and Y. Difference between X and Y.
    "definition",       # Define X. What is the definition of X?
    "reasoning",        # Why X? What causes X? What happens if?
    "summary",          # Summarize. Overview. Recap. Give me a summary.
    "notes",            # Generate notes. Notes for this lecture.
    "flashcards",       # Flashcards. Key terms. Study cards.
    "quiz",             # Quiz. Test me. Generate questions.
    "learning_path",    # Learning path. How to study. What to learn first.
]


@dataclass
class QueryPlan:
    """
    Full planner output for a single query.

    The retrieval_route field matches what workflow.run() consumes.
    All other fields are available to ContextBuilder and generation.
    """

    # Core routing — must match what the workflow expects.
    retrieval_route: RetrievalRoute

    # Classified intent.
    intent: QueryIntent = "factual_qa"

    # Whether the task is lecture-wide (summary, notes, etc.)
    # vs. point-specific (factual QA, definition).
    is_lecture_wide: bool = False

    # Answer style hint for the generation prompt.
    answer_style: Literal["concise", "detailed", "structured", "comparative"] = "concise"

    # Retrieval weight hints (0.0–1.0).  ContextBuilder may use these
    # to bias passage selection; retrieval architecture is not changed.
    graph_weight: float = 0.0
    vector_weight: float = 1.0

    # Graph traversal depth hint (passed to graph_retriever if used).
    graph_depth: int = 2

    # Top-K hint for vector retriever.
    top_k: int = 5

    # Character budget for final_context.
    context_budget: int = 4000

    # Whether visual context should be included.
    need_visual: bool = False
