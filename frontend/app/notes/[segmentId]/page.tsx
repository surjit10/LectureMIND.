// Notes — Screen 6
"use client";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { postNotes, getLecture } from "@/services/api";
import ErrorMessage from "@/components/common/ErrorMessage";
import LectureNav from "@/components/common/LectureNav";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import { ArrowLeft, FileText, Loader2, BookOpen } from "lucide-react";

export default function NotesPage() {
  const { segmentId } = useParams<{ segmentId: string }>();
  
  const { data: lectureData } = useQuery({
    queryKey: ["lecture", segmentId],
    queryFn: () => getLecture(segmentId),
  });

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["notes", segmentId],
    queryFn: () => postNotes(segmentId),
  });

  const displayName = lectureData?.display_name || lectureData?.title || segmentId;

  return (
    <div className="notes-shell">
      {/* ── Header ── */}
      <div className="notes-header">
        <Link href={`/lecture/${segmentId}`} className="notes-back-link">
          <ArrowLeft size={16} /> Back
        </Link>
        <div className="notes-title-row">
          <div className="notes-icon-box">
            <FileText size={24} />
          </div>
          <div>
            <h1 className="notes-title">Comprehensive Notes</h1>
            <p className="notes-subtitle flex items-center gap-2">
              <BookOpen size={14} /> {displayName}
            </p>
          </div>
        </div>
        <div className="notes-nav-wrapper">
          <LectureNav lectureId={data?.lecture_id || segmentId} segmentId={segmentId} />
        </div>
      </div>

      {/* ── Content ── */}
      <div className="notes-content-area">
        {isLoading && (
          <div className="notes-loading">
            <Loader2 size={32} className="spin" />
            <p>Generating structured notes...</p>
          </div>
        )}
        
        {error && <ErrorMessage message={error.message} onRetry={() => refetch()} />}
        
        {data && (
          <div className="notes-card">
            <div className="notes-md">
              <ReactMarkdown>
                {data.notes}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
