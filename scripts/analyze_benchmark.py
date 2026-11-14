"""Bootstrap confidence intervals + error analysis for one benchmark run (no LLM calls).

Usage:
    python scripts/analyze_benchmark.py --report <evaluation_report_*.json>
        --ablation <ablation per_question.jsonl> [--out <analysis.md>]

- 95% bootstrap CIs (10,000 resamples of questions, fixed seed) for the main
  metrics of the benchmark report.
- Error analysis: every answerable question is assigned exactly one outcome
  using the candidate pool / prompt context recorded by scripts/run_ablation.py
  (config ``full_pipeline``) and the generated answer from the report:
    ok                    gold chunk reached the prompt and the answer was not a refusal
    refused_with_evidence gold chunk reached the prompt but the model refused
    lost_after_retrieval  gold chunk was in the candidate pool but not in the prompt
    retrieval_miss        gold chunk never entered the candidate pool
"""
import argparse
import json
import random
import statistics as st
from collections import Counter
from pathlib import Path

REFUSAL = ("insufficient evidence", "not enough evidence", "cannot answer",
           "does not provide enough information", "no evidence", "unable to answer")
METRICS = ["post_rerank_hit_at_5", "post_rerank_mrr_at_5", "context_recall", "hit_at_5",
           "mrr_at_5", "answer_f1", "keyword_recall", "routing_accuracy", "total_latency"]


def bootstrap_ci(values, n=10_000, seed=0):
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text())
    results = report["results"]
    abl = {r["query"]: r for r in map(json.loads, Path(args.ablation).read_text().splitlines()) if r}

    lines = [f"# Benchmark analysis — `{Path(args.report).name}`", "",
             "## 95% bootstrap confidence intervals", "",
             "| Metric | Mean | 95% CI | n |", "|---|---:|---:|---:|"]
    cis = {}
    for m in METRICS:
        vals = [r["metrics"][m] for r in results if isinstance(r["metrics"].get(m), (int, float))]
        if not vals:
            continue
        lo, hi = bootstrap_ci(vals)
        cis[m] = {"mean": st.mean(vals), "ci95": [lo, hi], "n": len(vals)}
        lines.append(f"| {m} | {st.mean(vals):.4f} | [{lo:.4f}, {hi:.4f}] | {len(vals)} |")

    outcomes, rows = Counter(), []
    for r in results:
        if not r.get("answerable", True):
            continue
        a = abl.get(r["query"])
        if a is None:
            continue
        gold = set(a["expected_chunk_ids"])
        full = a["configs"]["full_pipeline"]
        refused = any(k in (r.get("answer_preview") or "").lower() for k in REFUSAL)
        in_ctx = bool(gold & set(full["context_ids"]))
        in_pool = bool(gold & set(full["ranked_ids"]))
        if in_ctx:
            outcome = "refused_with_evidence" if refused else "ok"
        else:
            outcome = "lost_after_retrieval" if in_pool else "retrieval_miss"
        outcomes[outcome] += 1
        if outcome != "ok":
            rank = next((i + 1 for i, c in enumerate(full["ranked_ids"]) if c in gold), None)
            rows.append((outcome, r["question_type"], rank, r["query"][:90]))

    total = sum(outcomes.values())
    lines += ["", "## Error analysis (answerable questions)", "",
              "| Outcome | Questions | Share |", "|---|---:|---:|"]
    for k in ("ok", "refused_with_evidence", "lost_after_retrieval", "retrieval_miss"):
        lines.append(f"| {k} | {outcomes[k]} | {outcomes[k] / total:.1%} |")
    lines += ["", "| Outcome | Type | Gold rank after rerank | Question |", "|---|---|---:|---|"]
    for o, t, rank, q in sorted(rows):
        lines.append(f"| {o} | {t} | {rank if rank else '—'} | {q} |")
    lines += ["", "`ok` means the evidence reached the model and it answered; answer correctness",
              "itself is not judged here (no LLM judge has been run)."]

    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        Path(args.out).with_suffix(".json").write_text(json.dumps({"ci95": cis, "outcomes": outcomes}, indent=2))
    print(text)


if __name__ == "__main__":
    main()
