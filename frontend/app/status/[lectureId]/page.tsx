// Processing Status — Screen 3
"use client";
import { useParams } from "next/navigation";
import { useStatusPolling } from "@/hooks/usePolling";
import StatusBadge from "@/components/common/StatusBadge";
import ErrorMessage from "@/components/common/ErrorMessage";
import Link from "next/link";

export default function StatusPage() {
  const { lectureId } = useParams<{ lectureId: string }>();
  const { data, isLoading, error } = useStatusPolling(lectureId, true);
  const isTerminal = data?.status === "READY" || data?.status === "FAILED";

  return (
    <div className="page">
      <h1>Processing Status</h1>
      <p className="subtitle">Lecture: {lectureId}</p>
      {isLoading && <div className="loader">Checking status...</div>}
      {error && <ErrorMessage message={error.message} />}
      {data && (
        <div className="status-panel">
          <StatusBadge status={data.status} />
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${data.progress * 100}%` }} />
          </div>
          <p className="progress-text">{Math.round(data.progress * 100)}% complete</p>
          {isTerminal && data.status === "READY" && (
            <div className="status-actions">
              <Link href={`/lecture/${lectureId}`} className="btn-primary">View Lecture →</Link>
              <Link href={`/chat/${lectureId}`} className="btn-secondary">Chat →</Link>
            </div>
          )}
          {isTerminal && data.status === "FAILED" && (
            <ErrorMessage message="Processing failed. Please re-upload." />
          )}
        </div>
      )}
    </div>
  );
}
