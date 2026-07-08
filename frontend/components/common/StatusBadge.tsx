// frontend/components/common/StatusBadge.tsx
"use client";
import type { PipelineStatus } from "@/types";

const STATUS_COLORS: Record<string, string> = {
  UPLOADED: "#6366f1", PROCESSING: "#f59e0b", SEGMENTING: "#f59e0b",
  EXTRACTING_ENTITIES: "#f59e0b", BUILDING_GRAPH: "#f59e0b",
  GENERATING_EMBEDDINGS: "#f59e0b", TRAINING_RERANKER: "#f59e0b",
  PACKAGING: "#f59e0b", READY: "#22c55e", FAILED: "#ef4444",
};

export default function StatusBadge({ status }: { status: PipelineStatus | string }) {
  const color = STATUS_COLORS[status] || "#94a3b8";
  return (
    <span className="status-badge" style={{ background: color }}>
      {status.replace(/_/g, " ")}
    </span>
  );
}
