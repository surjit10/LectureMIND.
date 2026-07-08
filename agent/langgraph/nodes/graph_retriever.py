# agent/langgraph/nodes/graph_retriever.py
# LangGraph node — wraps the Neo4j retriever.

from typing import Any, Dict, Optional

from retrieval.graph_retriever.neo4j_retriever import retrieve_graph


def graph_retriever_node(
    state: Dict[str, Any],
    driver: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    LangGraph node: populate state.graph_results from Neo4j.

    Only modifies graph_results. Does not touch other fields.
    """
    query = state.get("query", "")
    lecture_id = state.get("lecture_id", "")
    results = retrieve_graph(query, driver=driver, lecture_id=lecture_id if lecture_id else None)
    return {"graph_results": results}
