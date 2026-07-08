// components/settings/OfflineConfig.tsx
// Config panel shown when Inference Mode = "offline".
// Lets the user choose and save the local Ollama model.

"use client";
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getOllamaModels, patchSettings } from "@/services/api";
import { RefreshCw, CheckCircle, XCircle, HardDrive, Cpu } from "lucide-react";

const PRESET_MODELS = [
  "llama3.1:8b-instruct-q4_K_M",
  "qwen2.5:3b",
  "qwen2.5:7b",
  "mistral:7b",
  "phi3:mini",
  "gemma2:2b",
];

interface Props {
  currentModel: string;
  onSaved: () => void;
}

export default function OfflineConfig({ currentModel, onSaved }: Props) {
  const [model, setModel] = useState(currentModel);
  const [customModel, setCustomModel] = useState("");
  const [useCustom, setUseCustom] = useState(!PRESET_MODELS.includes(currentModel));

  const queryClient = useQueryClient();

  const { data: ollamaData, isLoading: modelsLoading, refetch } = useQuery({
    queryKey: ["ollama-models"],
    queryFn: getOllamaModels,
    retry: false,
  });

  const localModels = ollamaData?.models ?? [];

  const saveMutation = useMutation({
    mutationFn: () =>
      patchSettings({ local_model: useCustom ? customModel.trim() : model }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["ai-settings"] });
      onSaved();
    },
  });

  const activeModel = useCustom ? customModel.trim() : model;

  return (
    <div className="settings-form">
      {/* Local model selector */}
      <div className="settings-form-group">
        <div className="settings-label-row">
          <label className="settings-label">Local Model Selection</label>
          <button
            onClick={() => refetch()}
            disabled={modelsLoading}
            className="settings-action-link"
          >
            <RefreshCw size={14} className={modelsLoading ? "spin" : ""} />
            {modelsLoading ? "Loading..." : "Refresh"}
          </button>
        </div>

        {/* Preset list — only show if Ollama is reachable */}
        {localModels.length > 0 && !useCustom && (
          <div className="settings-options-grid">
            {localModels.map((m) => (
              <button
                key={m}
                onClick={() => setModel(m)}
                className={`settings-option-card ${model === m ? "selected" : ""}`}
              >
                <HardDrive size={16} />
                <span>{m}</span>
              </button>
            ))}
          </div>
        )}

        {/* Preset suggestions */}
        {localModels.length === 0 && !useCustom && (
          <div className="settings-options-grid cols-2">
            {PRESET_MODELS.map((m) => (
              <button
                key={m}
                onClick={() => setModel(m)}
                className={`settings-option-card ${model === m ? "selected" : ""}`}
              >
                <Cpu size={16} />
                <span>{m}</span>
              </button>
            ))}
          </div>
        )}

        {/* Custom model toggle & input */}
        <div className="mt-4">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={useCustom}
              onChange={(e) => setUseCustom(e.target.checked)}
            />
            <div className="settings-toggle-track">
              <div className="settings-toggle-thumb"></div>
            </div>
            <span>Use custom model name</span>
          </label>

          {useCustom && (
            <div className="mt-3">
              <input
                type="text"
                value={customModel}
                onChange={(e) => setCustomModel(e.target.value)}
                placeholder="e.g. llama3.1:70b"
                className="settings-input"
              />
              <p className="settings-input-hint">Ensure the model exists locally in Ollama.</p>
            </div>
          )}
        </div>
      </div>

      {/* Save */}
      <div className="settings-form-actions">
        {saveMutation.isError && (
          <span className="settings-validation error">
            <XCircle size={14} /> {(saveMutation.error as Error).message}
          </span>
        )}
        
        <button
          id="save-offline-settings"
          onClick={() => saveMutation.mutate()}
          disabled={saveMutation.isPending || !activeModel}
          className="btn-primary"
        >
          {saveMutation.isPending ? (
            <><RefreshCw size={16} className="spin" /> Saving...</>
          ) : saveMutation.isSuccess ? (
            <><CheckCircle size={16} /> Saved!</>
          ) : (
            "Save Settings"
          )}
        </button>
      </div>
    </div>
  );
}
