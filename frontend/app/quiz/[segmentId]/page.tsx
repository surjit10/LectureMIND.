// Quiz — Screen 8
"use client";
import { useState } from "react";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { postQuiz, getLecture } from "@/services/api";
import ErrorMessage from "@/components/common/ErrorMessage";
import LectureNav from "@/components/common/LectureNav";
import Link from "next/link";
import { ArrowLeft, HelpCircle, Loader2, BookOpen, CheckCircle2, XCircle } from "lucide-react";

export default function QuizPage() {
  const { segmentId } = useParams<{ segmentId: string }>();
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [showResults, setShowResults] = useState(false);

  const { data: lectureData } = useQuery({
    queryKey: ["lecture", segmentId],
    queryFn: () => getLecture(segmentId),
  });

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["quiz", segmentId],
    queryFn: () => postQuiz(segmentId),
  });

  const displayName = lectureData?.display_name || lectureData?.title || segmentId;
  const totalQuestions = data?.questions.length || 0;
  const answeredCount = Object.keys(answers).length;
  const isComplete = answeredCount === totalQuestions && totalQuestions > 0;
  
  let score = 0;
  if (showResults && data) {
    score = data.questions.filter((q, i) => answers[i] === q.correct).length;
  }

  const scorePercent = totalQuestions > 0 ? Math.round((score / totalQuestions) * 100) : 0;

  return (
    <div className="quiz-shell">
      {/* ── Header ── */}
      <div className="quiz-header">
        <Link href={`/lecture/${segmentId}`} className="quiz-back-link">
          <ArrowLeft size={16} /> Back
        </Link>
        <div className="quiz-title-row">
          <div className="quiz-icon-box">
            <HelpCircle size={24} />
          </div>
          <div className="flex-1">
            <h1 className="quiz-title">Knowledge Check</h1>
            <p className="quiz-subtitle flex items-center gap-2">
              <BookOpen size={14} /> {displayName}
            </p>
          </div>
        </div>
        <div className="quiz-nav-wrapper">
          <LectureNav lectureId={data?.lecture_id || segmentId} segmentId={segmentId} />
        </div>
      </div>

      {/* ── Content ── */}
      <div className="quiz-content-area">
        {isLoading && (
          <div className="quiz-loading">
            <Loader2 size={32} className="spin" />
            <p>Generating interactive quiz...</p>
          </div>
        )}

        {error && <ErrorMessage message={error.message} onRetry={() => refetch()} />}

        {data && totalQuestions > 0 && (
          <div className="quiz-main-layout">
            
            {/* Progress Bar */}
            <div className="quiz-progress-panel">
              <div className="flex justify-between text-sm text-gray-400 mb-2 font-medium">
                <span>Quiz Progress</span>
                <span>{answeredCount} / {totalQuestions} Answered</span>
              </div>
              <div className="quiz-progress-track">
                <div 
                  className="quiz-progress-fill" 
                  style={{ width: `${(answeredCount / totalQuestions) * 100}%` }} 
                />
              </div>
            </div>

            <div className="quiz-questions-list">
              {data.questions.map((q, i) => {
                const hasAnswered = answers[i] !== undefined;
                
                return (
                  <div key={i} className={`quiz-question-card ${hasAnswered ? 'answered' : ''}`}>
                    <div className="quiz-question-header">
                      <span className="quiz-q-number">Question {i + 1}</span>
                    </div>
                    <h3 className="quiz-q-text">{q.question}</h3>
                    
                    <div className="quiz-options-grid">
                      {(q.options || []).map((opt, j) => {
                        const isSelected = answers[i] === opt;
                        const isCorrect = opt === q.correct;
                        const isWrongSelected = showResults && isSelected && !isCorrect;
                        const isCorrectReveal = showResults && isCorrect;

                        let stateClass = "";
                        if (showResults) {
                          if (isCorrectReveal) stateClass = "correct";
                          else if (isWrongSelected) stateClass = "wrong";
                          else stateClass = "dimmed";
                        } else if (isSelected) {
                          stateClass = "selected";
                        }

                        return (
                          <div 
                            key={j} 
                            onClick={() => !showResults && setAnswers((p) => ({ ...p, [i]: opt }))}
                            className={`quiz-option-card ${stateClass} ${!showResults ? 'interactive' : ''}`}
                          >
                            <div className="quiz-opt-marker">
                              {String.fromCharCode(65 + j)}
                            </div>
                            <span className="quiz-opt-text">{opt}</span>
                            
                            {showResults && isCorrectReveal && <CheckCircle2 className="quiz-result-icon correct" size={20} />}
                            {showResults && isWrongSelected && <XCircle className="quiz-result-icon wrong" size={20} />}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Actions & Results */}
            <div className="quiz-footer">
              {!showResults ? (
                <button 
                  className="quiz-submit-btn" 
                  onClick={() => setShowResults(true)}
                  disabled={!isComplete}
                >
                  {isComplete ? "Submit Answers" : `Complete all ${totalQuestions} questions to submit`}
                </button>
              ) : (
                <div className="quiz-results-card">
                  <div className="quiz-score-circle" style={{ '--percent': scorePercent } as React.CSSProperties}>
                    <svg viewBox="0 0 36 36" className="circular-chart">
                      <path className="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                      <path className="circle" strokeDasharray={`${scorePercent}, 100`} d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                      <text x="18" y="20.35" className="percentage">{scorePercent}%</text>
                    </svg>
                  </div>
                  <div className="quiz-score-details">
                    <h3>Quiz Completed!</h3>
                    <p>You scored <strong>{score}</strong> out of <strong>{totalQuestions}</strong> correctly.</p>
                    <button className="btn-secondary mt-3" onClick={() => { setShowResults(false); setAnswers({}); }}>
                      Retake Quiz
                    </button>
                  </div>
                </div>
              )}
            </div>

          </div>
        )}
      </div>
    </div>
  );
}
