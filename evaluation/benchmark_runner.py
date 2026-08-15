# evaluation/benchmark_runner.py
# Runs the real QueryWorkflow against a live local stack (Qdrant + Neo4j +
# reranker + LLM backend) and computes grounded evaluation metrics.
#
# Injectable dependencies let the benchmark reuse the exact services the
# running app uses.  When a dependency is omitted:
#   - Qdrant client      -> created from LocalSettings (localhost:6333)
#   - Embedding model    -> preloaded bge-large-en-v1.5 (CPU)
#   - Reranker service   -> loaded from local_runtime/models/global_reranker/
#   - LLM backend        -> resolved via ProviderRegistry (online provider
#                           if configured, otherwise local Ollama)
#
# Usage:
#   runner = BenchmarkRunner(dataset_path, output_dir)
#   runner.setup()
#   runner.run()

import logging
import traceback
from typing import Any, Dict, List, Optional
from pathlib import Path

from .dataset_loader import DatasetLoader, BenchmarkSample
from .reports.report_generator import ReportGenerator

# Import metrics
from .metrics import planner_metrics, retrieval_metrics, reranker_metrics, answer_metrics, citation_metrics, latency_metrics

# Import the Workflow directly
from agent.langgraph.workflow import QueryWorkflow

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    def __init__(
        self,
        dataset_path: str,
        output_dir: str,
        qdrant_client: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        reranker_service: Optional[Any] = None,
        neo4j_driver: Optional[Any] = None,
        ollama_client: Optional[Any] = None,
        use_provider_registry_llm: bool = True,
    ):
        self.dataset_loader = DatasetLoader(dataset_path)
        self.output_dir = output_dir
        self.report_generator = ReportGenerator(output_dir)
        self.samples: List[BenchmarkSample] = []
        self.results: List[Dict[str, Any]] = []

        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.reranker_service = reranker_service
        self.neo4j_driver = neo4j_driver
        self.ollama_client = ollama_client
        self._use_provider_registry_llm = use_provider_registry_llm

        self.workflow = self._build_workflow()
        # Planner used to capture need_visual / route metadata for the record.
        # Stateless and deterministic; identical to the workflow's internal planner.
        from agent.dspy.planner import QueryPlanner
        self._planner = QueryPlanner()

    # ------------------------------------------------------------------
    # Dependency wiring
    # ------------------------------------------------------------------

    def _build_workflow(self) -> QueryWorkflow:
        """Construct a QueryWorkflow wired to the live local stack."""
        from config import LocalSettings

        # Qdrant client.
        qdrant_client = self.qdrant_client
        if qdrant_client is None:
            from qdrant_client import QdrantClient
            settings = LocalSettings()
            qdrant_client = QdrantClient(url=settings.QDRANT_URL)

        # Embedding model (bge-large-en-v1.5, 1024-dim) — preload once.
        embedding_model = self.embedding_model
        if embedding_model is None:
            from retrieval.vector_retriever.qdrant_retriever import preload_embedding_model
            embedding_model = preload_embedding_model(device="cpu")

        # Reranker service — load the global cross-encoder from disk,
        # exactly like app.py startup does.
        reranker_service = self.reranker_service
        if reranker_service is None:
            from local.loaders.reranker_loader import _load_cross_encoder
            from local.services.reranker_service import RerankerService
            from config import local_settings
            reranker_dir = Path(local_settings.GLOBAL_RERANKER_DIR)
            model = _load_cross_encoder(reranker_dir)
            reranker_service = RerankerService(model)

        # Neo4j driver (graph retrieval).  Optional: graph results are
        # non-fatal if the driver is unavailable (workflow guards this).
        neo4j_driver = self.neo4j_driver

        # LLM: if use_provider_registry_llm is True, leave ollama_client
        # unset so answer_generator_node resolves the active backend via
        # ProviderRegistry (online provider if configured).  Otherwise an
        # injected ollama_client is used directly.
        ollama_client = None if self._use_provider_registry_llm else self.ollama_client

        return QueryWorkflow(
            neo4j_driver=neo4j_driver,
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            reranker_service=reranker_service,
            ollama_client=ollama_client,
        )

    # ------------------------------------------------------------------
    # Benchmark lifecycle
    # ------------------------------------------------------------------

    def setup(self):
        """Prepares the benchmark runner by loading samples."""
        self.samples = self.dataset_loader.load()

    def run(self):
        """Executes the benchmark over all loaded samples."""
        logger.info(f"Starting benchmark run on {len(self.samples)} samples...")
        self.results = []
        for sample in self.samples:
            result = self._evaluate_sample(sample)
            self.results.append(result)
        logger.info("Benchmark run completed.")
        self.generate_report()
        return self.results

    def _evaluate_sample(self, sample: BenchmarkSample) -> Dict[str, Any]:
        """Evaluates a single sample against the workflow."""
        result_record = {
            "lecture_id": sample.lecture_id,
            "query": sample.query,
            "question_type": sample.question_type,
            "difficulty": sample.difficulty,
            "expected_route": sample.expected_route,
            "error": "",
            "metrics": {}
        }

        try:
            # 1. Execute workflow (real pipeline, real services).
            state = self.workflow.run(query=sample.query, lecture_id=sample.lecture_id)

            # 1b. Capture planner metadata (route + need_visual) for the record.
            plan = self._planner.plan_full(sample.query)
            actual_route = state.get("retrieval_route", "")
            # Use the enum NAME (graph_and_vector) so records match the
            # dataset's expected_route values (the enum VALUE is "graph+vector").
            actual_route_str = actual_route.name if hasattr(actual_route, "name") else str(actual_route)
            actual_need_visual = bool(getattr(plan, "need_visual", False))

            graph_results = state.get("graph_results", [])
            vector_results = state.get("vector_results", [])
            reranked_results = state.get("reranked_results", [])
            final_context = state.get("final_context", "")
            answer = state.get("answer", "")
            sources = state.get("sources", [])
            telemetry = state.get("telemetry", {})

            # Combined retrieval pool (pre-rerank).
            combined_retrieved = graph_results + vector_results
            # With hybrid retrieval enabled (BM25 + RRF), the reranker's pool
            # is the fused list, not the raw dense results — measure recall
            # against the pool that actually feeds the reranker.  When hybrid
            # is off, this reduces exactly to the dense-only pool.
            try:
                from config import LocalSettings
                if LocalSettings().ENABLE_HYBRID_RETRIEVAL and sample.lecture_id:
                    from retrieval.hybrid.bm25_retriever import bm25_search, rrf_fuse
                    bm25_retrieved = bm25_search(sample.query, sample.lecture_id, top_k=15)
                    fused = rrf_fuse(state.get("vector_results", []), bm25_retrieved, top_k=15)
                    if fused:
                        combined_retrieved = graph_results + fused
            except Exception:
                pass  # Hybrid is best-effort — fall back to the dense pool.
            retrieved_ids = []
            for r in combined_retrieved:
                payload = r.get("payload", r)
                cid = payload.get("chunk_id", r.get("chunk_id", ""))
                if cid:
                    retrieved_ids.append(cid)

            expected_ids = sample.expected_chunk_ids

            # 3. Compute metrics.
            metrics: Dict[str, Any] = {}

            metrics["routing_accuracy"] = planner_metrics.calculate_routing_accuracy(
                sample.expected_route, actual_route_str
            )
            # Visual routing: did the planner flag need_visual when the ground
            # truth requires visual context?
            metrics["visual_routing_accuracy"] = planner_metrics.calculate_routing_accuracy(
                "need_visual" if sample.need_visual else "no_visual",
                "need_visual" if actual_need_visual else "no_visual",
            )

            metrics["precision_at_5"] = retrieval_metrics.calculate_precision_at_k(expected_ids, retrieved_ids, 5)
            metrics["recall_at_5"] = retrieval_metrics.calculate_recall_at_k(expected_ids, retrieved_ids, 5)
            metrics["hit_at_5"] = retrieval_metrics.calculate_hit_at_k(expected_ids, retrieved_ids, 5)
            metrics["mrr"] = retrieval_metrics.calculate_mrr(expected_ids, retrieved_ids)
            metrics["ndcg_at_5"] = retrieval_metrics.calculate_ndcg(expected_ids, retrieved_ids, 5)

            metrics["ranking_quality"] = reranker_metrics.evaluate_ranking_quality(expected_ids, reranked_results)
            metrics["avg_cross_encoder_score"] = reranker_metrics.average_cross_encoder_score(reranked_results)

            metrics["answer_similarity"] = answer_metrics.calculate_answer_similarity(sample.ground_truth_answer, answer)
            metrics["answer_f1"] = answer_metrics.calculate_answer_f1(sample.ground_truth_answer, answer)
            keywords = getattr(sample, "keywords", []) or []
            metrics["keyword_recall"] = answer_metrics.calculate_keyword_recall(keywords, answer)
            metrics["context_length"] = answer_metrics.get_context_length(final_context)
            metrics["answer_length"] = answer_metrics.get_answer_length(answer)

            metrics["citation_coverage"] = citation_metrics.calculate_citation_coverage(expected_ids, sources)
            metrics["citation_completeness"] = citation_metrics.calculate_citation_completeness(
                final_context, sources, reranked_results
            )
            metrics["citation_count"] = citation_metrics.get_citation_count(sources)
            metrics["chunk_coverage"] = citation_metrics.get_chunk_coverage(sources, reranked_results)

            latencies = latency_metrics.extract_latencies(telemetry)
            metrics.update(latencies)

            result_record["metrics"] = metrics
            result_record["answer_preview"] = answer[:200]
            result_record["llm"] = {
                "provider": state.get("llm_metadata", {}).get("provider", ""),
                "model": state.get("llm_metadata", {}).get("model", ""),
            }

        except Exception as e:
            logger.error(f"Error evaluating sample '{sample.query}': {e}")
            logger.error(traceback.format_exc())
            result_record["error"] = str(e)

        return result_record

    def generate_report(self):
        """Generates the evaluation report."""
        self.report_generator.generate(self.results)
