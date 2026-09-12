// Chat — Screen 5
"use client";
import { useState, useRef, useEffect } from "react";
import { useParams } from "next/navigation";
import { useQuery, useMutation } from "@tanstack/react-query";
import { postQuery, getLecture } from "@/services/api";
import { useDeveloperMode } from "@/components/layout/Providers";
import PipelineTrace from "@/components/chat/PipelineTrace";
import { PrerequisiteNavigator } from "@/components/chat/PrerequisiteNavigator";
import ErrorMessage from "@/components/common/ErrorMessage";
import StatusBadge from "@/components/common/StatusBadge";
import LectureNav from "@/components/common/LectureNav";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import {
  ArrowLeft,
  Send,
  ChevronDown,
  ChevronRight,
  Clock,
  MessageSquare,
  BookOpen,
  GitBranch,
  Loader2,
} from "lucide-react";
import type { QueryResponse } from "@/types";

interface Message {
  role: "user" | "assistant";
  content: string;
  response?: QueryResponse;
}

export default function ChatPage() {
  const { lectureId } = useParams<{ lectureId: string }>();
  const { isDevMode } = useDeveloperMode();
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const { data: lectureData } = useQuery({
    queryKey: ["lecture", lectureId],
    queryFn: () => getLecture(lectureId),
  });

  const displayName =
    lectureData?.display_name || lectureData?.title || lectureId;

  const mutation = useMutation({
    mutationFn: (q: string) => postQuery(lectureId, q),
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, response: data },
      ]);
    },
  });

  const handleSend = () => {
    if (!input.trim() || mutation.isPending) return;
    setMessages((prev) => [...prev, { role: "user", content: input }]);
    mutation.mutate(input);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    // Auto-grow textarea
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px";
  };

  // Scroll to bottom whenever messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, mutation.isPending]);

  const duration = lectureData?.duration
    ? `${Math.round(lectureData.duration / 60)} min`
    : null;

  return (
    <div className="chat-shell">
      {/* ── Header ── */}
      <div className="chat-header">
        <Link href={`/lecture/${lectureId}`} className="chat-back-link">
          <ArrowLeft size={16} />
          Back
        </Link>

        {/* Lecture info card */}
        <div className="chat-lecture-card">
          <div className="chat-lecture-icon">
            <BookOpen size={20} />
          </div>
          <div className="chat-lecture-info">
            <span className="chat-lecture-name">{displayName}</span>
            <div className="chat-lecture-meta">
              {lectureData && <StatusBadge status={lectureData.status} />}
              {duration && (
                <span className="chat-meta-chip">
                  <Clock size={12} /> {duration}
                </span>
              )}
              <span className="chat-meta-chip">
                <MessageSquare size={12} /> {messages.filter((m) => m.role === "user").length} messages
              </span>
            </div>
          </div>
        </div>

        {/* Pill nav */}
        <div className="chat-nav-wrapper">
          <LectureNav lectureId={lectureId} />
        </div>
      </div>

      {/* ── Messages ── */}
      <div className="chat-messages-area">
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-empty-icon">
              <MessageSquare size={32} />
            </div>
            <h3>Ask anything about this lecture</h3>
            <p>
              Your questions are answered using GraphRAG — grounded in the
              actual lecture content.
            </p>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`chat-msg-row ${m.role}`}>
            {m.role === "assistant" && (
              <div className="chat-avatar assistant-avatar">AI</div>
            )}
            <div className={`chat-bubble-v2 ${m.role}`}>
              {m.role === "assistant" ? (
                <div className="chat-md">
                  <ReactMarkdown>{m.content}</ReactMarkdown>
                </div>
              ) : (
                <p>{m.content}</p>
              )}

              {/* Sources */}
              {m.response?.sources && m.response.sources.length > 0 && (
                <SourcesToggle sources={m.response.sources} />
              )}

              {/* Socratic Prerequisite Navigator */}
              {m.response?.prerequisites && m.response.prerequisites.length > 0 && (
                <PrerequisiteNavigator
                  prerequisites={m.response.prerequisites}
                  onSeek={(seconds) => {
                    const videoEl = document.querySelector("video");
                    if (videoEl) {
                      videoEl.currentTime = seconds;
                      videoEl.play().catch(() => {});
                    }
                  }}
                />
              )}

              {/* Graph path */}
              {m.response?.graph_path && m.response.graph_path.length > 0 && (
                <div className="chat-graph-row">
                  <GitBranch size={13} />
                  <span>{m.response.graph_path.join(" → ")}</span>
                </div>
              )}

              {/* Pipeline trace — Developer Mode only */}
              {isDevMode && m.response?.debug && (
                <PipelineTrace debug={m.response.debug} />
              )}
            </div>
            {m.role === "user" && (
              <div className="chat-avatar user-avatar">You</div>
            )}
          </div>
        ))}

        {mutation.isPending && (
          <div className="chat-msg-row assistant">
            <div className="chat-avatar assistant-avatar">AI</div>
            <div className="chat-bubble-v2 assistant chat-thinking">
              <Loader2 size={16} className="spin" />
              <span>Thinking…</span>
            </div>
          </div>
        )}

        {mutation.isError && (
          <div className="chat-error-row">
            <ErrorMessage message={mutation.error.message} />
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* ── Input ── */}
      <div className="chat-input-dock">
        <div className="chat-input-wrapper">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Ask about this lecture… (Enter to send, Shift+Enter for new line)"
            className="chat-textarea"
            rows={1}
            disabled={mutation.isPending}
          />
          <button
            onClick={handleSend}
            disabled={mutation.isPending || !input.trim()}
            className="chat-send-btn"
            aria-label="Send message"
          >
            {mutation.isPending ? (
              <Loader2 size={18} className="spin" />
            ) : (
              <Send size={18} />
            )}
          </button>
        </div>
        <p className="chat-hint">
          Grounded answers via GraphRAG · Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}

/* ── Sources toggle ── */
function SourcesToggle({
  sources,
}: {
  sources: import("@/types").Source[];
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="sources-panel">
      <button
        onClick={() => setOpen((o) => !o)}
        className="sources-toggle-btn"
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span>
          {sources.length} source{sources.length !== 1 ? "s" : ""}
        </span>
      </button>

      {open && (
        <div className="sources-list">
          {sources.map((s, j) => (
            <div key={j} className="source-item">
              <span className="source-chunk" title={s.chunk_id}>
                {s.chunk_id}
              </span>
              {s.timestamp && (
                <span className="source-ts">@ {s.timestamp}</span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
