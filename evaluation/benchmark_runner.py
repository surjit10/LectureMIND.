# evaluation/benchmark_runner.py
import logging
import traceback
from typing import Any, Dict, List
from pathlib import Path

from .dataset_loader import DatasetLoader, BenchmarkSample
from .reports.report_generator import ReportGenerator

# Import metrics
from .metrics import planner_metrics, retrieval_metrics, reranker_metrics, answer_metrics, citation_metrics, latency_metrics

# Import the Workflow directly
from agent.langgraph.workflow import QueryWorkflow

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    def __init__(self, dataset_path: str, output_dir: str):
        self.dataset_loader = DatasetLoader(dataset_path)
        self.output_dir = output_dir
        self.report_generator = ReportGenerator(output_dir)
        self.samples: List[BenchmarkSample] = []
        self.results: List[Dict[str, Any]] = []
        
        # We instantiate the workflow once for the benchmark.
        # In a real environment, dependencies (neo4j, qdrant, etc.) would be injected here.
        # For evaluation, we allow it to use the default components configured in workflow.py.
        self.workflow = QueryWorkflow()

    def setup(self):
        """Prepares the benchmark runner by loading samples."""
        self.samples = self.dataset_loader.load()

    def run(self):
        """Executes the benchmark over all loaded samples."""
        logger.info(f"Starting benchmark run on {len(self.samples)} samples...")
        
        for sample in self.samples:
            result = self._evaluate_sample(sample)
            self.results.append(result)
            
        logger.info("Benchmark run completed.")
        self.generate_report()

    def _evaluate_sample(self, sample: BenchmarkSample) -> Dict[str, Any]:
        """Evaluates a single sample against the workflow."""
        result_record = {
            "lecture_id": sample.lecture_id,
            "query": sample.query,
            "expected_route": sample.expected_route,
            "error": "",
            "metrics": {}
        }
        
        try:
            # 1. Execute workflow (DO NOT change inference)
            # This directly runs the single-pass pipeline and returns GraphState
            state = self.workflow.run(query=sample.query, lecture_id=sample.lecture_id)
            
            # 2. Extract values from state
            actual_route = state.get("retrieval_route", "")
            # stringify if it's an enum
            actual_route_str = actual_route.value if hasattr(actual_route, "value") else str(actual_route)
            
            graph_results = state.get("graph_results", [])
            vector_results = state.get("vector_results", [])
            reranked_results = state.get("reranked_results", [])
            final_context = state.get("final_context", "")
            answer = state.get("answer", "")
            sources = state.get("sources", [])
            telemetry = state.get("telemetry", {})  # Added in CP5
            
            # Helper to extract IDs for retrieval metrics
            # Combine graph and vector results as the "retrieved" pool before reranking
            combined_retrieved = graph_results + vector_results
            retrieved_ids = []
            for r in combined_retrieved:
                payload = r.get("payload", r)
                cid = payload.get("chunk_id", r.get("chunk_id", ""))
                if cid:
                    retrieved_ids.append(cid)
                    
            expected_ids = sample.expected_chunk_ids
            
            # 3. Compute Metrics
            metrics = {}
            
            # Planner Metrics
            metrics["routing_accuracy"] = planner_metrics.calculate_routing_accuracy(
                sample.expected_route, actual_route_str
            )
            
            # Retrieval Metrics
            metrics["precision_at_5"] = retrieval_metrics.calculate_precision_at_k(expected_ids, retrieved_ids, 5)
            metrics["recall_at_5"] = retrieval_metrics.calculate_recall_at_k(expected_ids, retrieved_ids, 5)
            metrics["hit_at_5"] = retrieval_metrics.calculate_hit_at_k(expected_ids, retrieved_ids, 5)
            metrics["mrr"] = retrieval_metrics.calculate_mrr(expected_ids, retrieved_ids)
            metrics["ndcg_at_5"] = retrieval_metrics.calculate_ndcg(expected_ids, retrieved_ids, 5)
            
            # Reranker Metrics
            metrics["ranking_quality"] = reranker_metrics.evaluate_ranking_quality(expected_ids, reranked_results)
            metrics["avg_cross_encoder_score"] = reranker_metrics.average_cross_encoder_score(reranked_results)
            
            # Answer Metrics
            metrics["answer_similarity"] = answer_metrics.calculate_answer_similarity(sample.ground_truth_answer, answer)
            metrics["keyword_recall"] = answer_metrics.calculate_keyword_recall([], answer) # No keywords in schema for now
            metrics["context_length"] = answer_metrics.get_context_length(final_context)
            metrics["answer_length"] = answer_metrics.get_answer_length(answer)
            
            # Citation Metrics
            metrics["citation_coverage"] = citation_metrics.calculate_citation_coverage(expected_ids, sources)
            metrics["citation_completeness"] = citation_metrics.calculate_citation_completeness(final_context, sources)
            metrics["citation_count"] = citation_metrics.get_citation_count(sources)
            metrics["chunk_coverage"] = citation_metrics.get_chunk_coverage(sources, reranked_results)
            
            # Latency Metrics
            latencies = latency_metrics.extract_latencies(telemetry)
            metrics.update(latencies)
            
            result_record["metrics"] = metrics
            
        except Exception as e:
            logger.error(f"Error evaluating sample '{sample.query}': {e}")
            logger.error(traceback.format_exc())
            result_record["error"] = str(e)
            
        return result_record

    def generate_report(self):
        """Generates the evaluation report."""
        self.report_generator.generate(self.results)
