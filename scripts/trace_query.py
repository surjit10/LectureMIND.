import sys
import os
sys.path.append(os.getcwd())

from agent.langgraph.workflow import QueryWorkflow
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from config import local_settings
from local.storage.registry_provider import get_registry
import logging

logging.basicConfig(level=logging.DEBUG)

lecture_id = get_registry().get_active_lecture_id()
if not lecture_id:
    lectures = get_registry().list_lectures()
    if lectures:
        lecture_id = lectures[0]["lecture_id"]

if not lecture_id:
    print("No active lecture.")
    sys.exit(1)

print(f"Using lecture_id: {lecture_id}")

driver = GraphDatabase.driver(local_settings.NEO4J_URI)
qdrant_client = QdrantClient(url=local_settings.QDRANT_URL)

workflow = QueryWorkflow(neo4j_driver=driver, qdrant_client=qdrant_client)

query = "explain the lecture"
state = workflow.run(query, lecture_id=lecture_id)

print("\n--- EXECUTION TRACE ---")
print("1. Planner")
from agent.dspy.planner import QueryPlanner
plan = QueryPlanner().plan_full(query)
print(f"route: {plan.retrieval_route}")
print(f"top_k: {plan.top_k}")
print(f"answer_style: {plan.answer_style}")
print(f"context_budget: {plan.context_budget}")
print(f"is_lecture_wide: {plan.is_lecture_wide}")

print("\n2. Vector Retrieval")
vector_results = state.get('vector_results', [])
print(f"Number of retrieved chunks: {len(vector_results)}")
for idx, r in enumerate(vector_results):
    payload = r.get('payload', {})
    ts = payload.get('timestamp', 0)
    transcript = payload.get('transcript', '')
    ocr = payload.get('ocr_text', '')
    vis = payload.get('visual_context', '')
    print(f"Chunk {idx}: chunk_id={payload.get('chunk_id')}, ts={ts}, score={r.get('score')}")
    print(f"  Transcript: {transcript[:100]}...")
    print(f"  OCR: {ocr[:100]}...")
    print(f"  Visual: {vis[:100]}...")

print("\n3. Graph Retrieval")
print(f"Number of graph results: {len(state.get('graph_results', []))}")

print("\n4. Reranker")
reranked_results = state.get('reranked_results', [])
print(f"Number of reranked chunks: {len(reranked_results)}")
for idx, r in enumerate(reranked_results):
    payload = r.get('payload', r)
    ts = payload.get('timestamp', 0)
    print(f"Reranked {idx}: chunk_id={payload.get('chunk_id')}, ts={ts}, rerank_score={r.get('rerank_score', r.get('score'))}")

print("\n4.5 Lecture-Wide Sampling (Simulated)")
from retrieval.context_builder import _sample_lecture_wide, _get_timestamp
unique = []
seen = set()
for r in reranked_results:
    cid = r.get('payload', r).get('chunk_id') or r.get('chunk_id')
    if cid not in seen:
        seen.add(cid)
        unique.append(r)
unique.sort(key=_get_timestamp)
sampled = _sample_lecture_wide(unique, n_buckets=8)
print(f"Number of sampled chunks: {len(sampled)}")
for idx, r in enumerate(sampled):
    payload = r.get('payload', r)
    print(f"Sampled {idx}: chunk_id={payload.get('chunk_id')}, ts={payload.get('timestamp', 0)}")

print("\n5. Context Builder")
final_context = state.get('final_context', '')
print(f"final_context length: {len(final_context)}")
print("First 1000 characters of final_context:")
print(final_context[:1000])
print("Last 1000 characters of final_context:")
print(final_context[-1000:])


print("\n6. Answer Generator")
print(f"Answer length: {len(state.get('answer', ''))}")
print(f"Answer preview: {state.get('answer', '')[:200]}")

from agent.langgraph.nodes.answer_generator import _resolve_backend, SYSTEM_PROMPT
backend = _resolve_backend(None, None, local_settings)
print(f"\nBackend: {type(backend)}")
try:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Test prompt"},
    ]
    res = backend.generate(messages)
    print(f"Backend test generate: '{res}'")
except Exception as e:
    print(f"Backend error: {e}")
