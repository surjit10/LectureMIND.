"""Compare two retrieval runs produced by scripts/compare_retrieval_run.py.

Usage: python scripts/compare_retrieval_analyze.py <old.jsonl> <new.jsonl> <out.json>
"""
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluation.metrics import retrieval_metrics as rm

OLD, NEW, OUT_JSON = (Path(p) for p in sys.argv[1:4])
PKG = REPO / "data/packages"


def load(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def norm_route(r):
    return (r or "").replace("graph+vector", "graph_and_vector")


def bench_metrics(recs):
    b = [r for r in recs if r["kind"] == "benchmark"]
    a = [r for r in b if r["answerable"]]
    m = lambda f: st.mean(f(r) for r in a)
    graph_routed = [r for r in b if r["route"] != "vector_only"]
    return {
        "n_questions": len(b), "n_answerable": len(a),
        "pre_rerank_hit_at_5": m(lambda r: rm.calculate_hit_at_k(r["expected_chunk_ids"], r["pool_ids"], 5)),
        "pre_rerank_mrr_at_5": m(lambda r: rm.calculate_mrr(r["expected_chunk_ids"], r["pool_ids"], k=5)),
        "post_rerank_hit_at_1": m(lambda r: rm.calculate_hit_at_k(r["expected_chunk_ids"], r["reranked_ids"], 1)),
        "post_rerank_hit_at_5": m(lambda r: rm.calculate_hit_at_k(r["expected_chunk_ids"], r["reranked_ids"], 5)),
        "post_rerank_recall_at_5": m(lambda r: rm.calculate_recall_at_k(r["expected_chunk_ids"], r["reranked_ids"], 5)),
        "post_rerank_mrr_at_5": m(lambda r: rm.calculate_mrr(r["expected_chunk_ids"], r["reranked_ids"], k=5)),
        "post_rerank_ndcg_at_5": m(lambda r: rm.calculate_ndcg(r["expected_chunk_ids"], r["reranked_ids"], 5)),
        "candidate_pool_recall": m(lambda r: rm.calculate_context_recall(r["expected_chunk_ids"], r["pool_ids"])),
        "context_recall": m(lambda r: rm.calculate_context_recall(r["expected_chunk_ids"], r["context_ids"])),
        "routing_accuracy": st.mean(norm_route(r["expected_route"]) == r["route"] for r in b),
        "graph_routed_questions": len(graph_routed),
        "mean_graph_lines_when_graph_routed": st.mean(r["graph_lines"] for r in graph_routed) if graph_routed else 0.0,
        "mean_retrieval_s": st.mean(r["retrieval_s"] for r in b),
        "mean_rerank_s": st.mean(r["rerank_s"] for r in b),
        "p95_retrieval_plus_rerank_s": sorted(r["retrieval_s"] + r["rerank_s"] for r in b)[int(0.95 * (len(b) - 1))],
    }


def lecture_duration(lid):
    chunks = json.loads((PKG / lid / "multimodal_chunks.json").read_text())
    return max(float(c.get("timestamp", 0) or 0) for c in chunks) or 1.0


def summary_metrics(recs):
    s = [r for r in recs if r["kind"] == "summary"]
    cov, dec = [], []
    for r in s:
        ts, dur = r["context_timestamps"], lecture_duration(r["lecture_id"])
        cov.append((max(ts) - min(ts)) / dur if ts else 0.0)
        dec.append(len({min(9, int(t / dur * 10)) for t in ts}))
    return {"n_prompts": len(s), "lecture_wide_rate": st.mean(r["is_lecture_wide"] for r in s),
            "mean_timeline_span": st.mean(cov), "min_timeline_span": min(cov),
            "mean_deciles_covered_of_10": st.mean(dec), "min_deciles_covered_of_10": min(dec)}


old, new = load(OLD), load(NEW)
result = {"old": {**bench_metrics(old), "summary": summary_metrics(old)},
          "new": {**bench_metrics(new), "summary": summary_metrics(new)}}

# Paired per-question comparison of post-rerank MRR@5 (answerable only).
ob = {r["idx"]: r for r in old if r["kind"] == "benchmark" and r["answerable"]}
nb = {r["idx"]: r for r in new if r["kind"] == "benchmark" and r["answerable"]}
better = worse = 0
changed = []
for i in ob:
    o = rm.calculate_mrr(ob[i]["expected_chunk_ids"], ob[i]["reranked_ids"], k=5)
    n = rm.calculate_mrr(nb[i]["expected_chunk_ids"], nb[i]["reranked_ids"], k=5)
    if n > o: better += 1
    if n < o: worse += 1
    if n != o or ob[i]["route"] != nb[i]["route"]:
        changed.append({"query": ob[i]["query"][:90], "route": f"{ob[i]['route']} -> {nb[i]['route']}",
                        "mrr5": f"{o:.2f} -> {n:.2f}"})
result["paired_post_rerank_mrr5"] = {"better": better, "worse": worse, "same": len(ob) - better - worse,
                                     "changed_questions": changed}
OUT_JSON.write_text(json.dumps(result, indent=2))

keys = [k for k in result["new"] if k != "summary"]
print(f"| Metric | Old code | New code |\n|---|---:|---:|")
for k in keys:
    o, n = result["old"][k], result["new"][k]
    fmt = (lambda v: f"{v:.4f}") if isinstance(o, float) else str
    print(f"| {k} | {fmt(o)} | {fmt(n)} |")
print("\nSummary prompts (lecture-wide coverage):")
for k in result["new"]["summary"]:
    o, n = result["old"]["summary"][k], result["new"]["summary"][k]
    print(f"| {k} | {o:.3f} | {n:.3f} |" if isinstance(o, float) else f"| {k} | {o} | {n} |")
print("\nPaired post-rerank MRR@5:", {k: v for k, v in result["paired_post_rerank_mrr5"].items() if k != "changed_questions"})
for c in changed:
    print("  ", c)
