// components/chat/PipelineTrace.tsx
// Developer-Mode only: renders the per-query pipeline trace returned by the
// backend in QueryResponse.debug. Reuses existing chat styling classes so the
// trace panel matches the sources panel. No student-facing behavior changes.

"use client";
import { useState } from "react";
import { ChevronDown, ChevronRight, GitBranch, Activity } from "lucide-react";
import type { QueryDebug } from "@/types";

function ms(value?: number): string {
  if (value === undefined || value === null) return "—";
  return `${(value * 1000).toFixed(1)} ms`;
}

function fmtScore(value?: number): string {
  if (value === undefined || value === null) return "—";
  return value.toFixed(4);
}

export default function PipelineTrace({ debug }: { debug: QueryDebug }) {
  const [open, setOpen] = useState(true);

  const telemetry = debug.telemetry ?? {};
  const stage = debug.stage_counts;

  return (
    <div className="sources-panel">
      <button
        onClick={() => setOpen((o) => !o)}
        className="sources-toggle-btn"
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <Activity size={13} />
        <span>Pipeline trace</span>
      </button>

      {open && (
        <div className="sources-list trace-list">
          {/* Route + planner decision */}
          <div className="trace-block">
            <span className="trace-label">Route</span>
            <span className="trace-value">{debug.retrieval_route || "—"}</span>
          </div>
          {debug.plan && (
            <div className="trace-block">
              <span className="trace-label">Planner</span>
              <span className="trace-value">
                intent={debug.plan.intent ?? "?"}
                {debug.plan.is_lecture_wide ? " · lecture-wide" : ""} · style=
                {debug.plan.answer_style ?? "?"} · top_k={debug.plan.top_k ?? "?"}
                {" "}budget={debug.plan.context_budget ?? "?"}
                {debug.plan.need_visual ? " · visual" : ""}
              </span>
            </div>
          )}

          {/* Stage funnel */}
          {stage && (
            <div className="trace-block">
              <span className="trace-label">Stages</span>
              <span className="trace-value">
                {stage.vector} vector → {stage.graph} graph → {stage.reranked} reranked
              </span>
            </div>
          )}

          {/* Graph path */}
          {debug.graph_path && debug.graph_path.length > 0 && (
            <div className="trace-block">
              <span className="trace-label"><GitBranch size={11} /> Path</span>
              <span className="trace-value">{debug.graph_path.join(" → ")}</span>
            </div>
          )}

          {/* Latencies */}
          <div className="trace-block">
            <span className="trace-label">Timings</span>
            <span className="trace-value">
              plan {ms(telemetry.planner_latency)} · retrieve {ms(telemetry.retrieval_latency)} · rerank {ms(telemetry.reranker_latency)} · generate {ms(telemetry.generation_latency)} · total {ms(telemetry.total_latency)}
            </span>
          </div>

          {debug.final_context_chars !== undefined && (
            <div className="trace-block">
              <span className="trace-label">Context</span>
              <span className="trace-value">{debug.final_context_chars} chars fed to LLM</span>
            </div>
          )}

          {/* Top vector scores */}
          {debug.vector_results && debug.vector_results.length > 0 && (
            <details className="trace-details">
              <summary>Vector scores ({debug.vector_results.length})</summary>
              {debug.vector_results.map((r, i) => (
                <div key={i} className="trace-row">
                  <span className="source-chunk" title={r.chunk_id}>{r.chunk_id}</span>
                  <span className="trace-ts">{fmtScore(r.score)}</span>
                </div>
              ))}
            </details>
          )}

          {/* Rerank scores */}
          {debug.reranked_results && debug.reranked_results.length > 0 && (
            <details className="trace-details">
              <summary>Reranked top chunks ({debug.reranked_results.length})</summary>
              {debug.reranked_results.map((r, i) => (
                <div key={i} className="trace-row">
                  <span className="source-chunk" title={r.chunk_id}>{r.chunk_id}</span>
                  <span className="trace-ts">{fmtScore(r.rerank_score)}</span>
                </div>
              ))}
            </details>
          )}

          {/* Graph traversal rows */}
          {debug.graph_results && debug.graph_results.length > 0 && (
            <details className="trace-details">
              <summary>Graph traversal ({debug.graph_results.length})</summary>
              {debug.graph_results.map((r, i) => (
                <div key={i} className="trace-row">
                  <span className="source-chunk">
                    {r.start}{r.related ? ` → ${r.related}` : ""}
                  </span>
                  <span className="trace-ts">{[...(r.rel_types ?? [])].join(",") || "—"}</span>
                </div>
              ))}
            </details>
          )}
        </div>
      )}
    </div>
  );
}
