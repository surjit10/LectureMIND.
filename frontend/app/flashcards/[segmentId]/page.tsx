// Flashcards — Screen 7
"use client";
import { useState } from "react";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { postFlashcards, getLecture } from "@/services/api";
import ErrorMessage from "@/components/common/ErrorMessage";
import LectureNav from "@/components/common/LectureNav";
import Link from "next/link";
import { ArrowLeft, Layers, Loader2, BookOpen } from "lucide-react";

export default function FlashcardsPage() {
  const { segmentId } = useParams<{ segmentId: string }>();
  const [flipped, setFlipped] = useState<Record<number, boolean>>({});

  const { data: lectureData } = useQuery({
    queryKey: ["lecture", segmentId],
    queryFn: () => getLecture(segmentId),
  });

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["flashcards", segmentId],
    queryFn: () => postFlashcards(segmentId),
  });

  const displayName = lectureData?.display_name || lectureData?.title || segmentId;
  const totalFlipped = Object.values(flipped).filter(Boolean).length;
  const totalCards = data?.flashcards.length || 0;
  const progressPercent = totalCards > 0 ? (totalFlipped / totalCards) * 100 : 0;

  return (
    <div className="flashcards-shell">
      {/* ── Header ── */}
      <div className="flashcards-header">
        <Link href={`/lecture/${segmentId}`} className="flashcards-back-link">
          <ArrowLeft size={16} /> Back
        </Link>
        <div className="flashcards-title-row">
          <div className="flashcards-icon-box">
            <Layers size={24} />
          </div>
          <div className="flex-1">
            <h1 className="flashcards-title">Flashcards</h1>
            <p className="flashcards-subtitle flex items-center gap-2">
              <BookOpen size={14} /> {displayName}
            </p>
          </div>
        </div>
        <div className="flashcards-nav-wrapper">
          <LectureNav lectureId={data?.lecture_id || segmentId} segmentId={segmentId} />
        </div>
      </div>

      {/* ── Content ── */}
      <div className="flashcards-content-area">
        {isLoading && (
          <div className="flashcards-loading">
            <Loader2 size={32} className="spin" />
            <p>Generating flashcards...</p>
          </div>
        )}

        {error && <ErrorMessage message={error.message} onRetry={() => refetch()} />}

        {data && data.flashcards.length > 0 && (
          <>
            <div className="flashcards-progress-bar">
              <div 
                className="flashcards-progress-fill" 
                style={{ width: `${progressPercent}%` }} 
              />
              <span className="flashcards-progress-text">
                {totalFlipped} of {totalCards} flipped
              </span>
            </div>

            <div className="flashcards-grid-v2">
              {data.flashcards.map((fc, i) => (
                <div 
                  key={i} 
                  className={`flashcard-v2 ${flipped[i] ? "flipped" : ""}`}
                  onClick={() => setFlipped((p) => ({ ...p, [i]: !p[i] }))}
                >
                  <div className="flashcard-v2-inner">
                    <div className="flashcard-v2-front">
                      <span className="flashcard-v2-number">Q{i + 1}</span>
                      <p>{fc.question}</p>
                      <span className="flashcard-v2-hint">Click to reveal</span>
                    </div>
                    <div className="flashcard-v2-back">
                      <span className="flashcard-v2-number">A{i + 1}</span>
                      <p>{fc.answer}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
