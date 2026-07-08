// Learning Path — Screen 9
"use client";
import { useState } from "react";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { postLearningPath, getLecture } from "@/services/api";
import ErrorMessage from "@/components/common/ErrorMessage";
import LectureNav from "@/components/common/LectureNav";
import Link from "next/link";
import { ArrowLeft, Map, Loader2, BookOpen, CheckCircle2, Circle, Target, Milestone } from "lucide-react";

export default function LearningPathPage() {
  const { lectureId } = useParams<{ lectureId: string }>();
  
  // Track which steps the user has completed locally
  const [completedSteps, setCompletedSteps] = useState<Record<number, boolean>>({});

  const { data: lectureData } = useQuery({
    queryKey: ["lecture", lectureId],
    queryFn: () => getLecture(lectureId),
  });

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["learningPath", lectureId],
    queryFn: () => postLearningPath(lectureId),
  });

  const displayName = lectureData?.display_name || lectureData?.title || lectureId;
  const totalSteps = data?.path.length || 0;
  const completedCount = Object.values(completedSteps).filter(Boolean).length;
  const progressPercent = totalSteps > 0 ? (completedCount / totalSteps) * 100 : 0;

  const toggleStep = (index: number) => {
    setCompletedSteps(prev => ({ ...prev, [index]: !prev[index] }));
  };

  return (
    <div className="path-shell">
      {/* ── Header ── */}
      <div className="path-header">
        <Link href={`/lecture/${lectureId}`} className="path-back-link">
          <ArrowLeft size={16} /> Back
        </Link>
        <div className="path-title-row">
          <div className="path-icon-box">
            <Map size={24} />
          </div>
          <div className="flex-1">
            <h1 className="path-title">Learning Path</h1>
            <p className="path-subtitle flex items-center gap-2">
              <BookOpen size={14} /> {displayName}
            </p>
          </div>
        </div>
        <div className="path-nav-wrapper">
          <LectureNav lectureId={lectureId} />
        </div>
      </div>

      {/* ── Content ── */}
      <div className="path-content-area">
        {isLoading && (
          <div className="path-loading">
            <Loader2 size={32} className="spin" />
            <p>Building your personalized learning path...</p>
          </div>
        )}

        {error && <ErrorMessage message={error.message} onRetry={() => refetch()} />}

        {data && totalSteps > 0 && (
          <div className="path-main-layout">
            
            {/* Progress Header */}
            <div className="path-progress-card">
              <div className="path-progress-header">
                <div className="flex items-center gap-2 text-pink-500 font-bold">
                  <Target size={20} /> Course Progress
                </div>
                <span className="text-gray-400 font-medium">{completedCount} of {totalSteps} modules completed</span>
              </div>
              <div className="path-progress-track">
                <div 
                  className="path-progress-fill" 
                  style={{ width: `${progressPercent}%` }} 
                />
              </div>
            </div>

            {/* Timeline */}
            <div className="path-timeline-container">
              {data.path.map((step, i) => {
                const isCompleted = !!completedSteps[i];
                const isNext = !isCompleted && (i === 0 || !!completedSteps[i - 1]);
                
                let stateClass = "";
                if (isCompleted) stateClass = "completed";
                else if (isNext) stateClass = "active";

                return (
                  <div key={i} className={`path-node ${stateClass}`}>
                    
                    {/* Connector line (not on last item) */}
                    {i < totalSteps - 1 && (
                      <div className="path-connector" />
                    )}

                    {/* Left: Icon/Milestone */}
                    <div className="path-milestone">
                      <div className="path-milestone-circle">
                        {isCompleted ? (
                          <CheckCircle2 size={24} className="text-white" />
                        ) : isNext ? (
                          <Milestone size={22} className="text-white" />
                        ) : (
                          <Circle size={16} className="text-gray-500" />
                        )}
                      </div>
                    </div>

                    {/* Right: Content Card */}
                    <div className="path-card" onClick={() => toggleStep(i)}>
                      <div className="path-card-header">
                        <span className="path-module-badge">Module {i + 1}</span>
                        {isCompleted && <span className="path-status-badge">Done</span>}
                        {isNext && <span className="path-status-badge next">Up Next</span>}
                      </div>
                      <h3 className="path-card-title">{step.title}</h3>
                      <p className="path-card-desc">{step.description}</p>
                      
                      <div className="path-card-action">
                        {isCompleted ? "Mark as unread" : "Mark as complete"}
                      </div>
                    </div>

                  </div>
                );
              })}
            </div>

          </div>
        )}

        {data && totalSteps === 0 && (
          <div className="path-empty">
            <Map size={48} className="text-gray-600 mb-4" />
            <h3>No learning path available</h3>
            <p className="text-gray-400 max-w-sm text-center mt-2">
              We couldn't generate a structured path for this lecture. Try re-processing the knowledge package.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
