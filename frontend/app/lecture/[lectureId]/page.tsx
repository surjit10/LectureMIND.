// Lecture Details — Screen 4
"use client";
import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getLecture, renameLecture } from "@/services/api";
import StatusBadge from "@/components/common/StatusBadge";
import ErrorMessage from "@/components/common/ErrorMessage";
import Link from "next/link";
import { useDeveloperMode } from "@/components/layout/Providers";
import { Edit2, Check, X, Clock, User, Globe, ArrowLeft, Terminal } from "lucide-react";
import LectureNav from "@/components/common/LectureNav";

export default function LecturePage() {
  const { lectureId } = useParams<{ lectureId: string }>();
  const queryClient = useQueryClient();
  const { isDevMode } = useDeveloperMode();
  
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState("");

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["lecture", lectureId],
    queryFn: () => getLecture(lectureId),
  });

  useEffect(() => {
    if (data) {
      setEditName(data.display_name || data.title || data.lecture_id);
    }
  }, [data]);

  const renameMutation = useMutation({
    mutationFn: (newName: string) => renameLecture(lectureId, newName),
    onSuccess: () => {
      setIsEditing(false);
      queryClient.invalidateQueries({ queryKey: ["lecture", lectureId] });
      queryClient.invalidateQueries({ queryKey: ["lectures"] });
    },
  });

  const handleRename = () => {
    if (editName.trim() && editName !== (data?.display_name || data?.title)) {
      renameMutation.mutate(editName);
    } else {
      setIsEditing(false);
    }
  };

  return (
    <div className="lecture-details-shell">
      <div className="mb-6">
        <Link href="/" className="lecture-back-link">
          <ArrowLeft size={16} /> Back to Library
        </Link>
      </div>
      
      {isLoading && (
        <div className="lecture-loading-skeleton">
          <div className="h-40 bg-gray-800 rounded-xl animate-pulse"></div>
          <div className="h-16 bg-gray-800 rounded-xl animate-pulse"></div>
        </div>
      )}
      
      {error && <ErrorMessage message={error.message} onRetry={() => refetch()} />}
      
      {data && (
        <div className="lecture-details-content">
          
          {/* Main Hero Card */}
          <div className="lecture-hero-card">
            <div className="lecture-hero-glow" />
            
            <div className="lecture-hero-inner">
              <div className="flex items-center gap-3 mb-4">
                <StatusBadge status={data.status} />
                {data.course_name && (
                  <span className="lecture-course-badge">{data.course_name}</span>
                )}
              </div>
              
              <div className="lecture-title-row">
                {isEditing ? (
                  <div className="lecture-edit-group">
                    <input 
                      type="text" 
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleRename()}
                      autoFocus
                      disabled={renameMutation.isPending}
                      className="settings-input text-2xl font-bold py-2"
                    />
                    <button 
                      onClick={handleRename} 
                      disabled={renameMutation.isPending} 
                      className="lecture-action-btn success"
                    >
                      <Check size={20} />
                    </button>
                    <button 
                      onClick={() => { setIsEditing(false); setEditName(data.display_name || data.title || ""); }} 
                      disabled={renameMutation.isPending} 
                      className="lecture-action-btn danger"
                    >
                      <X size={20} />
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center gap-3 group">
                    <h1 className="lecture-hero-title">
                      {data.display_name || data.title || data.lecture_id}
                    </h1>
                    <button 
                      onClick={() => setIsEditing(true)} 
                      className="lecture-edit-trigger"
                      title="Rename package"
                    >
                      <Edit2 size={18} />
                    </button>
                  </div>
                )}
              </div>
              
              <div className="lecture-meta-grid">
                {data.speaker && (
                  <div className="lecture-meta-item">
                    <div className="meta-icon"><User size={14} /></div>
                    <span><span className="meta-label">Speaker:</span> {data.speaker}</span>
                  </div>
                )}
                {data.duration !== undefined && data.duration > 0 && (
                  <div className="lecture-meta-item">
                    <div className="meta-icon"><Clock size={14} /></div>
                    <span><span className="meta-label">Duration:</span> {Math.round(data.duration / 60)} mins</span>
                  </div>
                )}
                {data.language && (
                  <div className="lecture-meta-item">
                    <div className="meta-icon"><Globe size={14} /></div>
                    <span><span className="meta-label">Language:</span> {data.language.toUpperCase()}</span>
                  </div>
                )}
              </div>
              
              {data.description && (
                <p className="lecture-hero-desc">{data.description}</p>
              )}
            </div>
          </div>
          
          {/* Navigation Pill */}
          <div className="lecture-nav-container">
            <LectureNav lectureId={lectureId} />
          </div>

          {/* Dev Tools */}
          {isDevMode && (
            <div className="lecture-dev-card">
              <div className="lecture-dev-header">
                <Terminal size={16} /> Technical Details
              </div>
              <div className="lecture-dev-grid">
                <div>
                  <div className="dev-label">Lecture ID</div>
                  <div className="dev-value monospace">{data.lecture_id}</div>
                </div>
                <div className="flex gap-12">
                  <div>
                    <div className="dev-label">Segments</div>
                    <div className="dev-value text-xl">{data.segment_count}</div>
                  </div>
                  <div>
                    <div className="dev-label">Chunks</div>
                    <div className="dev-value text-xl">{data.chunk_count}</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
