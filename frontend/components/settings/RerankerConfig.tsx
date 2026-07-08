// components/settings/RerankerConfig.tsx
// Global Reranker Configuration Panel.
// Allows uploading, validating, and hot-reloading a custom CrossEncoder model.
// Uses only existing LectureMind design system tokens and CSS classes.

"use client";
import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getGlobalRerankerStatus,
  uploadGlobalReranker,
  reloadGlobalReranker,
  deleteGlobalReranker,
  patchGlobalRerankerSettings,
} from "@/services/api";
import {
  RefreshCw,
  Trash2,
  CheckCircle,
  XCircle,
  UploadCloud,
  Package,
  HardDrive,
  Loader2,
  AlertTriangle,
  Info,
  Edit2,
  Check,
  X,
} from "lucide-react";

// ─────────────────────────────────────────────────────────────
// Stage list for upload progress
// ─────────────────────────────────────────────────────────────
const STAGES = ["Uploading", "Extracting", "Validating", "Installing", "Reloading"];

export default function RerankerConfig() {
  const queryClient = useQueryClient();
  const [errorMsg, setErrorMsg] = useState("");
  const [activeStage, setActiveStage] = useState(-1); // -1 = idle
  const [showConfirm, setShowConfirm] = useState(false);
  const [isEditingMaxUpload, setIsEditingMaxUpload] = useState(false);
  const [newMaxUploadMb, setNewMaxUploadMb] = useState("2000");

  const invalidate = () =>
    queryClient.invalidateQueries({ queryKey: ["reranker-status"] });

  const { data: status, isLoading } = useQuery({
    queryKey: ["reranker-status"],
    queryFn: getGlobalRerankerStatus,
  });

  // ── Upload mutation ──────────────────────────────────────
  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      setErrorMsg("");
      // Animate through stages while the single HTTP call runs
      setActiveStage(0);
      const tick = (i: number) =>
        setTimeout(() => setActiveStage(i), i * 1200);
      [1, 2, 3].forEach((i) => tick(i));
      return uploadGlobalReranker(file);
    },
    onSuccess: () => {
      setActiveStage(4); // Reloading → done
      setTimeout(() => {
        setActiveStage(-1);
        invalidate();
      }, 1500);
    },
    onError: (err: Error) => {
      setActiveStage(-1);
      setErrorMsg(err.message);
    },
  });

  // ── Reload mutation ──────────────────────────────────────
  const reloadMutation = useMutation({
    mutationFn: reloadGlobalReranker,
    onSuccess: () => invalidate(),
    onError: (err: Error) => setErrorMsg("Reload failed: " + err.message),
  });

  // ── Delete mutation ──────────────────────────────────────
  const deleteMutation = useMutation({
    mutationFn: deleteGlobalReranker,
    onSuccess: () => {
      setShowConfirm(false);
      invalidate();
    },
    onError: (err: Error) => {
      setShowConfirm(false);
      setErrorMsg("Delete failed: " + err.message);
    },
  });

  // ── Patch settings mutation ────────────────────────────────
  const patchSettingsMutation = useMutation({
    mutationFn: patchGlobalRerankerSettings,
    onSuccess: () => {
      setIsEditingMaxUpload(false);
      invalidate();
    },
    onError: (err: Error) => setErrorMsg("Update failed: " + err.message),
  });

  // ── Dropzone ─────────────────────────────────────────────
  const isWorking =
    uploadMutation.isPending ||
    reloadMutation.isPending ||
    deleteMutation.isPending;

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;
      if (!file.name.endsWith(".zip")) {
        setErrorMsg(
          "Invalid file type. Please upload a .zip file containing a HuggingFace CrossEncoder model."
        );
        return;
      }
      uploadMutation.mutate(file);
    },
    [uploadMutation]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/zip": [".zip"] },
    maxFiles: 1,
    disabled: isWorking,
  });

  const uploadSuccess = uploadMutation.isSuccess && activeStage === -1;
  const maxUploadMb = status?.max_upload_size_mb ?? 2000;
  const maxUploadLabel =
    maxUploadMb >= 1000 ? `${(maxUploadMb / 1000).toFixed(0)} GB` : `${maxUploadMb} MB`;

  if (isLoading) {
    return (
      <div className="settings-form">
        <div style={{ height: "120px", background: "rgba(255,255,255,0.04)", borderRadius: "var(--radius)", animation: "pulse 1.5s infinite" }} />
        <div style={{ height: "160px", background: "rgba(255,255,255,0.04)", borderRadius: "var(--radius)", animation: "pulse 1.5s infinite" }} />
      </div>
    );
  }

  const hasCustomModel = status?.custom_model === true;
  const isLoaded = status?.loaded === true;

  return (
    <div className="settings-form">

      {/* ── Current Model ─────────────────────────────────── */}
      <div className="settings-form-group">
        <span className="settings-section-label">Current Model</span>

        {/* Info rows */}
        <div style={{
          background: "rgba(255,255,255,0.02)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          overflow: "hidden",
        }}>
          {/* Model name + status badge */}
          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "1rem 1.5rem",
            borderBottom: "1px solid var(--border)",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <div style={{
                display: "flex", alignItems: "center", justifyContent: "center",
                width: "36px", height: "36px", borderRadius: "10px",
                background: "var(--accent-gradient)", color: "#fff", flexShrink: 0,
              }}>
                <Package size={18} />
              </div>
              <div>
                <p style={{ fontWeight: 600, color: "var(--text-primary)", margin: 0, fontSize: "1rem" }}>
                  {status?.model_name ?? "Not Installed"}
                </p>
                <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", margin: 0 }}>
                  {status?.backend ?? "CrossEncoder"}
                </p>
              </div>
            </div>

            {/* Status badge — matches provider-active-badge style */}
            <div style={{
              display: "inline-flex", alignItems: "center", gap: "0.35rem",
              fontSize: "0.75rem", fontWeight: 600, padding: "4px 10px",
              borderRadius: "var(--radius-full)",
              background: isLoaded ? "rgba(16,185,129,0.12)" : "rgba(245,158,11,0.12)",
              color: isLoaded ? "#34d399" : "#fbbf24",
              border: `1px solid ${isLoaded ? "rgba(16,185,129,0.3)" : "rgba(245,158,11,0.3)"}`,
            }}>
              <span style={{
                width: "6px", height: "6px", borderRadius: "50%",
                background: isLoaded ? "#34d399" : "#fbbf24",
                display: "inline-block",
              }} />
              {isLoaded ? "Loaded" : "Not Loaded"}
            </div>
          </div>

          {/* Metadata rows */}
          <div style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "0.75rem 1.5rem", borderBottom: "1px solid rgba(255,255,255,0.03)"
          }}>
            <span style={{ fontSize: "0.9rem", color: "var(--text-muted)" }}>Disk Usage</span>
            <span style={{ fontSize: "0.9rem", color: "var(--text-secondary)", fontWeight: 500 }}>
              {status?.disk_size_mb ? `${status.disk_size_mb} MB` : "—"}
            </span>
          </div>

          <div style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "0.75rem 1.5rem", borderBottom: "1px solid rgba(255,255,255,0.03)",
            minHeight: "44px"
          }}>
            <span style={{ fontSize: "0.9rem", color: "var(--text-muted)" }}>Maximum Upload</span>
            
            {isEditingMaxUpload ? (
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <input 
                  type="number"
                  value={newMaxUploadMb}
                  onChange={(e) => setNewMaxUploadMb(e.target.value)}
                  style={{
                    width: "80px",
                    background: "rgba(255,255,255,0.05)",
                    border: "1px solid rgba(255,255,255,0.15)",
                    color: "white",
                    borderRadius: "4px",
                    padding: "2px 6px",
                    fontSize: "0.85rem",
                    outline: "none"
                  }}
                  disabled={patchSettingsMutation.isPending}
                />
                <span style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginRight: "0.25rem" }}>MB</span>
                <button 
                  onClick={() => {
                    const val = parseInt(newMaxUploadMb, 10);
                    if (!isNaN(val)) patchSettingsMutation.mutate(val);
                  }}
                  disabled={patchSettingsMutation.isPending}
                  style={{ background: "transparent", border: "none", color: "#34d399", cursor: "pointer", padding: "2px", display: "flex" }}
                >
                  {patchSettingsMutation.isPending ? <Loader2 size={14} className="spin" /> : <Check size={14} />}
                </button>
                <button 
                  onClick={() => setIsEditingMaxUpload(false)}
                  disabled={patchSettingsMutation.isPending}
                  style={{ background: "transparent", border: "none", color: "#f87171", cursor: "pointer", padding: "2px", display: "flex" }}
                >
                  <X size={14} />
                </button>
              </div>
            ) : (
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span style={{ fontSize: "0.9rem", color: "var(--text-secondary)", fontWeight: 500 }}>
                  {maxUploadLabel}
                </span>
                <button 
                  onClick={() => {
                    setNewMaxUploadMb(String(maxUploadMb));
                    setIsEditingMaxUpload(true);
                  }}
                  style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer", display: "flex" }}
                  title="Edit Max Upload Size"
                >
                  <Edit2 size={13} style={{ opacity: 0.7 }} />
                </button>
              </div>
            )}
          </div>

          <div style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "0.75rem 1.5rem", borderBottom: "1px solid rgba(255,255,255,0.03)"
          }}>
            <span style={{ fontSize: "0.9rem", color: "var(--text-muted)" }}>Last Updated</span>
            <span style={{ fontSize: "0.9rem", color: "var(--text-secondary)", fontWeight: 500 }}>
              {status?.last_updated ?? "—"}
            </span>
          </div>

          {/* Action buttons */}
          <div style={{
            display: "flex", gap: "0.75rem", padding: "1rem 1.5rem",
            background: "rgba(255,255,255,0.01)",
          }}>
            <button
              onClick={() => reloadMutation.mutate()}
              disabled={isWorking}
              className="btn-secondary"
              style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem" }}
            >
              {reloadMutation.isPending
                ? <><Loader2 size={15} className="spin" /> Reloading...</>
                : reloadMutation.isSuccess
                ? <><CheckCircle size={15} /> Reloaded!</>
                : <><RefreshCw size={15} /> Reload Model</>}
            </button>

            <button
              onClick={() => setShowConfirm(true)}
              disabled={isWorking || !hasCustomModel}
              className="settings-danger-btn"
              style={{
                flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
                gap: "0.5rem", opacity: (!hasCustomModel || isWorking) ? 0.4 : 1,
                cursor: (!hasCustomModel || isWorking) ? "not-allowed" : "pointer",
              }}
            >
              {deleteMutation.isPending
                ? <><Loader2 size={15} className="spin" /> Deleting...</>
                : <><Trash2 size={15} /> Delete Model</>}
            </button>
          </div>
        </div>

        {/* Empty-state message when no custom model */}
        {!hasCustomModel && (
          <div style={{
            display: "flex", alignItems: "flex-start", gap: "0.75rem",
            padding: "0.85rem 1rem",
            background: "rgba(139,92,246,0.06)",
            border: "1px solid rgba(139,92,246,0.15)",
            borderRadius: "var(--radius-sm)",
            marginTop: "0.75rem",
          }}>
            <Info size={16} style={{ color: "var(--accent-hover)", flexShrink: 0, marginTop: "0.1rem" }} />
            <p style={{ fontSize: "0.88rem", color: "var(--text-muted)", margin: 0, lineHeight: 1.5 }}>
              No custom reranker installed. LectureMind is using the default built-in model.
              Upload a CrossEncoder ZIP below to replace it.
            </p>
          </div>
        )}
      </div>

      {/* ── Delete Confirmation Modal ─────────────────────── */}
      {showConfirm && (
        <div className="settings-modal-overlay" onClick={() => setShowConfirm(false)}>
          <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
            <div className="settings-modal-icon">
              <AlertTriangle size={28} />
            </div>
            <h2 className="settings-modal-title">Delete Custom Reranker?</h2>
            <p className="settings-modal-desc" style={{ color: "var(--text-secondary)" }}>
              This will permanently remove the installed custom reranker from local storage.
              LectureMind will automatically restore and reload the default built-in model.
            </p>
            <div className="settings-modal-actions">
              <button className="btn-secondary" onClick={() => setShowConfirm(false)}>
                Cancel
              </button>
              <button
                className="btn-danger"
                onClick={() => deleteMutation.mutate()}
                disabled={deleteMutation.isPending}
              >
                {deleteMutation.isPending ? <><Loader2 size={15} className="spin" /> Deleting...</> : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Error Banner ──────────────────────────────────── */}
      {errorMsg && (
        <div style={{
          display: "flex", alignItems: "flex-start", gap: "0.85rem",
          padding: "1rem 1.25rem",
          background: "var(--error-bg)",
          border: "1px solid rgba(239,68,68,0.25)",
          borderRadius: "var(--radius)",
        }}>
          <XCircle size={18} style={{ color: "var(--error)", flexShrink: 0, marginTop: "0.1rem" }} />
          <div style={{ flex: 1 }}>
            <p style={{ fontWeight: 600, color: "#f87171", margin: "0 0 0.2rem 0", fontSize: "0.95rem" }}>
              Installation Failed
            </p>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", margin: 0 }}>{errorMsg}</p>
          </div>
          <button
            onClick={() => setErrorMsg("")}
            style={{ background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", fontSize: "0.85rem", fontWeight: 500 }}
          >
            Dismiss
          </button>
        </div>
      )}

      {/* ── Upload Progress ───────────────────────────────── */}
      {activeStage >= 0 && (
        <div style={{
          padding: "1.5rem",
          background: "rgba(255,255,255,0.02)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
        }}>
          <div style={{
            display: "flex", alignItems: "center", gap: "0.6rem",
            marginBottom: "1rem",
          }}>
            <Loader2 size={16} style={{ color: "var(--accent)", animation: "spin 1s linear infinite" }} />
            <span style={{ fontWeight: 600, color: "var(--text-primary)", fontSize: "0.95rem" }}>
              {STAGES[activeStage]}...
            </span>
          </div>

          {/* Stage steps */}
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            {STAGES.map((stage, i) => {
              const done = i < activeStage;
              const active = i === activeStage;
              return (
                <div key={stage} style={{
                  display: "flex", alignItems: "center", gap: "0.6rem",
                  opacity: i > activeStage ? 0.35 : 1,
                }}>
                  {done
                    ? <CheckCircle size={15} style={{ color: "#34d399", flexShrink: 0 }} />
                    : <div style={{
                        width: 15, height: 15, borderRadius: "50%", flexShrink: 0,
                        border: `2px solid ${active ? "var(--accent)" : "var(--border)"}`,
                        background: active ? "rgba(139,92,246,0.2)" : "transparent",
                      }} />}
                  <span style={{
                    fontSize: "0.875rem",
                    color: done ? "#34d399" : active ? "var(--text-primary)" : "var(--text-muted)",
                    fontWeight: active ? 600 : 400,
                  }}>
                    {stage}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Upload Success ────────────────────────────────── */}
      {uploadSuccess && (
        <div className="settings-banner success" style={{ alignItems: "flex-start" }}>
          <CheckCircle size={20} style={{ flexShrink: 0, marginTop: "0.1rem" }} />
          <div>
            <p style={{ fontWeight: 600, margin: "0 0 0.2rem 0" }}>Custom reranker installed successfully.</p>
            <p style={{ fontWeight: 400, opacity: 0.8, margin: 0 }}>Model loaded into memory. Ready for inference.</p>
          </div>
        </div>
      )}

      {/* ── Upload Zone ───────────────────────────────────── */}
      <div className="settings-form-group">
        <span className="settings-section-label">Upload Custom Reranker</span>

        <div
          {...getRootProps()}
          style={{
            padding: "3rem 2rem",
            border: `2px dashed ${isDragActive ? "var(--accent)" : "rgba(255,255,255,0.12)"}`,
            borderRadius: "var(--radius-lg)",
            background: isDragActive
              ? "rgba(139,92,246,0.08)"
              : isWorking
              ? "rgba(255,255,255,0.01)"
              : "rgba(255,255,255,0.02)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            textAlign: "center",
            cursor: isWorking ? "not-allowed" : "pointer",
            transition: "var(--transition)",
            opacity: isWorking ? 0.5 : 1,
            transform: isDragActive ? "scale(1.01)" : "scale(1)",
            boxShadow: isDragActive ? "var(--shadow-glow)" : "none",
          }}
        >
          <input {...getInputProps()} />

          {/* Icon */}
          <div style={{
            width: 64, height: 64, borderRadius: "var(--radius)",
            background: isDragActive ? "rgba(139,92,246,0.2)" : "rgba(255,255,255,0.05)",
            display: "flex", alignItems: "center", justifyContent: "center",
            marginBottom: "1.25rem", transition: "var(--transition)",
          }}>
            <UploadCloud size={32} style={{ color: isDragActive ? "var(--accent-hover)" : "var(--text-muted)" }} />
          </div>

          {/* Text */}
          <p style={{ fontSize: "1.15rem", fontWeight: 600, color: "var(--text-primary)", margin: "0 0 0.4rem 0" }}>
            {isDragActive ? "Drop ZIP here" : "Drag & Drop your Global Reranker ZIP"}
          </p>
          <p style={{ fontSize: "0.9rem", color: "var(--text-muted)", margin: "0 0 1.5rem 0" }}>
            or
          </p>

          {/* Browse button */}
          <button
            type="button"
            disabled={isWorking}
            style={{
              padding: "0.55rem 1.5rem",
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: "var(--radius-full)",
              color: "var(--text-primary)",
              fontWeight: 600,
              fontSize: "0.9rem",
              cursor: isWorking ? "not-allowed" : "pointer",
              transition: "var(--transition)",
              marginBottom: "1.75rem",
            }}
            onMouseEnter={(e) => { if (!isWorking) (e.target as HTMLElement).style.borderColor = "rgba(255,255,255,0.3)"; }}
            onMouseLeave={(e) => { (e.target as HTMLElement).style.borderColor = "rgba(255,255,255,0.12)"; }}
          >
            Browse Files
          </button>

          {/* Metadata pills */}
          <div style={{
            display: "flex", gap: "1.5rem", alignItems: "center",
            paddingTop: "1.25rem",
            borderTop: "1px solid rgba(255,255,255,0.05)",
            width: "100%", maxWidth: "360px", justifyContent: "center",
          }}>
            <div>
              <p style={{ fontSize: "0.72rem", textTransform: "uppercase", letterSpacing: "0.8px", color: "var(--text-muted)", margin: "0 0 0.2rem 0" }}>Format</p>
              <p style={{ fontSize: "0.88rem", fontWeight: 600, color: "var(--text-secondary)", margin: 0 }}>ZIP</p>
            </div>
            <div style={{ width: 1, height: 28, background: "var(--border)" }} />
            <div>
              <p style={{ fontSize: "0.72rem", textTransform: "uppercase", letterSpacing: "0.8px", color: "var(--text-muted)", margin: "0 0 0.2rem 0" }}>Max Size</p>
              <p style={{ fontSize: "0.88rem", fontWeight: 600, color: "var(--text-secondary)", margin: 0 }}>{maxUploadLabel}</p>
            </div>
            <div style={{ width: 1, height: 28, background: "var(--border)" }} />
            <div>
              <p style={{ fontSize: "0.72rem", textTransform: "uppercase", letterSpacing: "0.8px", color: "var(--text-muted)", margin: "0 0 0.2rem 0" }}>Required</p>
              <p style={{ fontSize: "0.88rem", fontWeight: 600, color: "var(--text-secondary)", margin: 0 }}>CrossEncoder</p>
            </div>
          </div>
        </div>

        <p className="settings-help-text" style={{ marginTop: "0.6rem" }}>
          ZIP must contain <code style={{ color: "var(--accent-hover)", fontSize: "0.85em" }}>config.json</code>,{" "}
          <code style={{ color: "var(--accent-hover)", fontSize: "0.85em" }}>tokenizer.json</code>, and model weights{" "}
          (<code style={{ color: "var(--accent-hover)", fontSize: "0.85em" }}>model.safetensors</code> or{" "}
          <code style={{ color: "var(--accent-hover)", fontSize: "0.85em" }}>pytorch_model.bin</code>).
        </p>
      </div>
    </div>
  );
}
