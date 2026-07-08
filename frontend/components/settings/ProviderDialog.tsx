// components/settings/ProviderDialog.tsx
// Modal for adding a new online provider or editing an existing one.
// Implements intelligent model discovery via API key testing.

"use client";
import { useState, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { addProvider, updateProvider, testNewProvider } from "@/services/api";
import type { Provider, ProviderType } from "@/types";
import { X, Loader2, CheckCircle, XCircle, Search } from "lucide-react";

const PROVIDERS: { value: ProviderType; label: string; placeholder: string; baseUrl?: string }[] = [
  { value: "google_gemini", label: "Google Gemini", placeholder: "gemini-2.5-flash" },
  { value: "openai", label: "OpenAI", placeholder: "gpt-4o" },
  { value: "groq", label: "Groq", placeholder: "llama-3.3-70b-versatile" },
  { value: "openrouter", label: "OpenRouter", placeholder: "meta-llama/llama-3.1-70b-instruct" },
  { value: "anthropic", label: "Anthropic", placeholder: "claude-sonnet-4-5" },
  { value: "custom", label: "Custom (OpenAI-compatible)", placeholder: "your-model-name", baseUrl: "https://your-endpoint/v1" },
];

interface Props {
  editTarget?: Provider | null;
  onClose: () => void;
}

export default function ProviderDialog({ editTarget, onClose }: Props) {
  const isEdit = Boolean(editTarget);
  const queryClient = useQueryClient();

  const [provider, setProvider] = useState<ProviderType>("google_gemini");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [baseUrl, setBaseUrl] = useState("");

  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);
  const [useCustomModel, setUseCustomModel] = useState(false);

  useEffect(() => {
    if (editTarget) {
      setProvider(editTarget.provider);
      setModel(editTarget.model);
      setDisplayName(editTarget.display_name);
      setBaseUrl(editTarget.base_url || "");
      setUseCustomModel(true);
    }
  }, [editTarget]);

  const selectedMeta = PROVIDERS.find((p) => p.value === provider);

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["ai-settings"] });
    onClose();
  };

  const addMutation = useMutation({
    mutationFn: () =>
      addProvider({
        provider,
        model: model.trim(),
        api_key: apiKey.trim(),
        display_name: displayName.trim() || undefined,
        base_url: baseUrl.trim() || undefined,
      }),
    onSuccess: invalidate,
  });

  const editMutation = useMutation({
    mutationFn: () =>
      updateProvider(editTarget!.id, {
        model: model.trim() || undefined,
        api_key: apiKey.trim() || undefined,
        display_name: displayName.trim() || undefined,
        base_url: baseUrl.trim() || undefined,
      }),
    onSuccess: invalidate,
  });

  const testMutation = useMutation({
    mutationFn: () =>
      testNewProvider({
        provider,
        api_key: apiKey.trim(),
        base_url: baseUrl.trim() || undefined,
      }),
    onSuccess: (data) => {
      setTestResult({ success: data.success, message: data.message });
      if (data.success && data.models && data.models.length > 0) {
        setAvailableModels(data.models);
        setModel(data.models[0]); // auto-select first
        setUseCustomModel(false);
      } else if (data.success) {
        setAvailableModels([]);
        setUseCustomModel(true);
      }
    },
    onError: (err: Error) => {
      setTestResult({ success: false, message: err.message });
    }
  });

  const handleSubmit = () => {
    if (!model.trim()) return;
    if (!isEdit && !apiKey.trim()) return;
    if (isEdit) {
      editMutation.mutate();
    } else {
      addMutation.mutate();
    }
  };

  const handleTest = () => {
    if (!apiKey.trim()) return;
    setTestResult(null);
    testMutation.mutate();
  };

  const isPending = addMutation.isPending || editMutation.isPending;
  const error = addMutation.error || editMutation.error;
  
  const canSubmit = model.trim() && (isEdit || apiKey.trim()) && !isPending && !testMutation.isPending;

  return (
    <div className="settings-modal-overlay">
      <div className="settings-modal provider-modal">
        {/* Header */}
        <div className="settings-modal-header">
          <h2 className="settings-modal-title">{isEdit ? "Edit Provider" : "Add Provider"}</h2>
          <button onClick={onClose} className="settings-modal-close">
            <X size={20} />
          </button>
        </div>

        <div className="settings-form">
          {/* Provider select */}
          {!isEdit && (
            <div className="settings-form-group">
              <label className="settings-label">Cloud Provider</label>
              <select
                id="provider-select"
                value={provider}
                onChange={(e) => {
                  setProvider(e.target.value as ProviderType);
                  setModel("");
                  setBaseUrl("");
                  setAvailableModels([]);
                  setTestResult(null);
                }}
                className="settings-select"
              >
                {PROVIDERS.map((p) => (
                  <option key={p.value} value={p.value}>{p.label}</option>
                ))}
              </select>
            </div>
          )}

          {/* Base URL */}
          {(provider === "custom" || isEdit) && (
            <div className="settings-form-group">
              <label className="settings-label">
                Base URL <span className="settings-label-optional">(optional)</span>
              </label>
              <input
                id="provider-base-url"
                type="url"
                value={baseUrl}
                onChange={(e) => { setBaseUrl(e.target.value); setTestResult(null); }}
                placeholder={selectedMeta?.baseUrl ?? "https://your-endpoint/v1"}
                className="settings-input"
              />
            </div>
          )}

          {/* API Key */}
          <div className="settings-form-group">
            <div className="settings-label-row">
              <label className="settings-label">
                API Key {isEdit && <span className="settings-label-optional">(leave blank to keep current)</span>}
              </label>
            </div>
            <div className="settings-input-action-group">
              <input
                id="provider-api-key"
                type="password"
                value={apiKey}
                onChange={(e) => { setApiKey(e.target.value); setTestResult(null); }}
                placeholder={isEdit ? "Enter new key to replace..." : "sk-..."}
                autoComplete="new-password"
                className="settings-input monospace"
              />
              <button
                onClick={handleTest}
                disabled={!apiKey.trim() || testMutation.isPending}
                className="settings-inline-btn"
              >
                {testMutation.isPending ? <Loader2 size={16} className="spin" /> : "Test Key"}
              </button>
            </div>
            
            {/* Test Connection Result */}
            {testResult && (
              <div className={`settings-validation ${testResult.success ? "success" : "error"} mt-2`}>
                {testResult.success ? <CheckCircle size={14} /> : <XCircle size={14} />}
                {testResult.message}
              </div>
            )}
          </div>

          {/* Model Selection */}
          <div className="settings-form-group divider-top pt-4">
            <div className="settings-label-row">
              <label className="settings-label">Target Model</label>
              {!useCustomModel && availableModels.length > 0 && (
                <button 
                  onClick={() => { setUseCustomModel(true); setModel(""); }}
                  className="settings-action-link"
                >
                  Enter manually
                </button>
              )}
            </div>
            
            {!useCustomModel && availableModels.length > 0 ? (
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="settings-select"
              >
                {availableModels.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            ) : (
              <div className="space-y-2">
                <input
                  id="provider-model"
                  type="text"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder={selectedMeta?.placeholder ?? "model-name"}
                  className="settings-input"
                />
                {availableModels.length > 0 && (
                  <button 
                    onClick={() => { setUseCustomModel(false); setModel(availableModels[0]); }}
                    className="settings-action-link"
                  >
                    View discovered models
                  </button>
                )}
                {!isEdit && availableModels.length === 0 && !useCustomModel && (
                   <p className="settings-input-hint flex items-center gap-1.5 mt-2">
                     <Search size={14} /> Test connection to auto-discover models
                   </p>
                )}
              </div>
            )}
          </div>

          {/* Display name */}
          <div className="settings-form-group divider-top pt-4">
            <label className="settings-label">Display Name <span className="settings-label-optional">(optional)</span></label>
            <input
              id="provider-display-name"
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder={selectedMeta?.label ?? "My Provider"}
              className="settings-input"
            />
          </div>

        </div>

        {error && (
          <div className="settings-validation error mt-4">
            <XCircle size={14} /> {(error as Error).message}
          </div>
        )}

        {/* Actions */}
        <div className="settings-modal-actions mt-8">
          <button onClick={onClose} className="btn-secondary">Cancel</button>
          <button
            id="provider-save-btn"
            onClick={handleSubmit}
            disabled={!canSubmit}
            className="btn-primary"
          >
            {isPending ? <><Loader2 size={16} className="spin" /> Saving...</> : "Save Provider"}
          </button>
        </div>
      </div>
    </div>
  );
}
