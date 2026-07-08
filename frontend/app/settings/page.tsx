// Settings — Screen 10
"use client";
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getSettings, patchSettings, resetAllLectures } from "@/services/api";
import ErrorMessage from "@/components/common/ErrorMessage";
import { useDeveloperMode } from "@/components/layout/Providers";
import InferenceModeSelector from "@/components/settings/InferenceModeSelector";
import OfflineConfig from "@/components/settings/OfflineConfig";
import OnlineConfig from "@/components/settings/OnlineConfig";
import RerankerConfig from "@/components/settings/RerankerConfig";
import { Settings as SettingsIcon, Database, AlertTriangle, CheckCircle, Package } from "lucide-react";
import { useRouter } from "next/navigation";
import type { InferenceMode } from "@/types";

export default function SettingsPage() {
  const { isDevMode } = useDeveloperMode();
  const [resetModal, setResetModal] = useState(false);
  const [savedBanner, setSavedBanner] = useState(false);
  const router = useRouter();
  const queryClient = useQueryClient();

  const { data: settings, isLoading, error, refetch } = useQuery({
    queryKey: ["ai-settings"],
    queryFn: getSettings,
  });

  const modeMutation = useMutation({
    mutationFn: (mode: InferenceMode) => patchSettings({ inference_mode: mode }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["ai-settings"] }),
  });

  const resetMutation = useMutation({
    mutationFn: resetAllLectures,
    onSuccess: () => {
      setResetModal(false);
      router.push("/");
    },
  });

  const handleSaved = () => {
    setSavedBanner(true);
    setTimeout(() => setSavedBanner(false), 3000);
  };

  return (
    <div className="settings-shell">
      {/* ── Page Header ── */}
      <div className="settings-header">
        <div className="settings-title-row">
          <div className="settings-icon-box">
            <SettingsIcon size={28} />
          </div>
          <div>
            <h1 className="settings-title">Settings</h1>
            <p className="settings-subtitle">Configure your LectureMind preferences</p>
          </div>
        </div>
      </div>

      {/* ── Content ── */}
      <div className="settings-content-area">
        {/* Saved banner */}
        {savedBanner && (
          <div className="settings-banner success">
            <CheckCircle size={18} /> 
            <span>Settings saved successfully.</span>
          </div>
        )}

        {/* AI Configuration card */}
        <div className="settings-card">
          <div className="settings-card-header">
            <Database size={22} className="text-primary" />
            <h2>AI Configuration</h2>
          </div>

          <div className="settings-card-body">
            {isLoading && (
              <div className="flex flex-col gap-4">
                <div className="h-24 bg-gray-800 rounded-xl animate-pulse" />
                <div className="h-32 bg-gray-800 rounded-xl animate-pulse" />
              </div>
            )}

            {error && <ErrorMessage message={error.message} onRetry={() => refetch()} />}

            {settings && (
              <div className="settings-section-group">
                {/* Section 1: Mode selector */}
                <div className="settings-section">
                  <InferenceModeSelector
                    value={settings.inference_mode}
                    onChange={(mode) => modeMutation.mutate(mode)}
                    disabled={modeMutation.isPending}
                  />
                </div>

                <div className="settings-divider" />

                {/* Section 2: Config Panel */}
                <div className="settings-section">
                  {settings.inference_mode === "offline" ? (
                    <div className="settings-sub-panel">
                      <h3>Local Model Configuration</h3>
                      <p className="settings-help-text">Configure your local Ollama connection.</p>
                      <OfflineConfig currentModel={settings.local_model} onSaved={handleSaved} />
                    </div>
                  ) : (
                    <div className="settings-sub-panel">
                      <h3>Cloud Providers</h3>
                      <p className="settings-help-text">Manage your API keys for cloud-based LLM providers.</p>
                      <OnlineConfig settings={settings} />
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Global Reranker card */}
        <div className="settings-card">
          <div className="settings-card-header">
            <Package size={22} className="text-primary" />
            <h2>Global Reranker Configuration</h2>
          </div>
          <div className="settings-card-body">
            <RerankerConfig />
          </div>
        </div>

        {/* Developer Tools */}
        {isDevMode && (
          <div className="settings-card danger-card">
            <div className="settings-card-header text-red-500">
              <AlertTriangle size={22} />
              <h2>Developer Tools</h2>
            </div>
            <div className="settings-card-body">
              <p className="text-red-400 mb-6 text-sm bg-red-500/10 p-3 rounded-lg border border-red-500/20">
                Dangerous actions for development and debugging purposes only.
              </p>
              <div className="settings-danger-action">
                <div className="danger-text">
                  <h3>Reset All Packages</h3>
                  <p>Permanently delete all imported packages, metadata, and indexes.</p>
                </div>
                <button
                  onClick={() => setResetModal(true)}
                  className="settings-danger-btn"
                >
                  Reset Everything
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Reset Modal */}
      {resetModal && (
        <div className="settings-modal-overlay">
          <div className="settings-modal">
            <div className="settings-modal-icon">
              <AlertTriangle size={28} />
            </div>
            <h2 className="settings-modal-title">Delete ALL imported packages?</h2>
            <p className="settings-modal-desc">
              This removes everything from the backend (metadata, DB indexes, caches, and stored
              files). This action cannot be undone.
            </p>
            <div className="settings-modal-actions">
              <button
                onClick={() => setResetModal(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={() => resetMutation.mutate()}
                disabled={resetMutation.isPending}
                className="btn-danger"
              >
                {resetMutation.isPending ? "Deleting..." : "Delete All"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
