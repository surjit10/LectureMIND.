// components/settings/ProviderCard.tsx
// Displays a single configured online provider.
// Shows masked API key, model, and actions (Set Active, Edit, Delete).

"use client";
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteProvider, testProvider, patchSettings } from "@/services/api";
import type { Provider } from "@/types";
import { Edit2, Trash2, CheckCircle, XCircle, Loader2, KeyRound, Globe2 } from "lucide-react";

const PROVIDER_LABELS: Record<string, string> = {
  google_gemini: "Google Gemini",
  openai: "OpenAI",
  groq: "Groq",
  openrouter: "OpenRouter",
  anthropic: "Anthropic",
  custom: "Custom Provider",
};

interface Props {
  provider: Provider;
  isActive: boolean;
  onEdit: (provider: Provider) => void;
}

export default function ProviderCard({ provider, isActive, onEdit }: Props) {
  const queryClient = useQueryClient();
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ["ai-settings"] });

  const activateMutation = useMutation({
    mutationFn: () => patchSettings({ active_provider_id: provider.id }),
    onSuccess: invalidate,
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteProvider(provider.id),
    onSuccess: invalidate,
  });

  const testMutation = useMutation({
    mutationFn: () => testProvider(provider.id),
    onSuccess: (data) => setTestResult(data),
    onError: (err: Error) => setTestResult({ success: false, message: err.message }),
  });

  return (
    <div className={`provider-card ${isActive ? 'active' : ''}`} onClick={() => !isActive && activateMutation.mutate()}>
      {/* Background glow when active */}
      <div className="provider-card-bg"></div>

      <div className="provider-card-header">
        <div className="provider-card-identity">
          <div className="provider-card-icon">
            <Globe2 size={20} />
          </div>
          <div className="provider-card-title-stack">
            <h3>{provider.display_name || PROVIDER_LABELS[provider.provider] || provider.provider}</h3>
            <span>{provider.model}</span>
          </div>
        </div>
        
        {isActive && (
          <div className="provider-active-badge">
            <CheckCircle size={14} /> Active
          </div>
        )}
      </div>

      <div className="provider-card-body">
        <div className="provider-key-row">
          <KeyRound size={14} className="text-gray-500" />
          <span>••••••••••••••••{provider.api_key.slice(-4)}</span>
        </div>
        {provider.base_url && (
          <div className="provider-url-row" title={provider.base_url}>
            Base URL: {provider.base_url}
          </div>
        )}
      </div>

      <div className="provider-card-footer" onClick={(e) => e.stopPropagation()}>
        <button
          onClick={() => { setTestResult(null); testMutation.mutate(); }}
          disabled={testMutation.isPending}
          className="provider-action-btn test"
        >
          {testMutation.isPending ? <Loader2 size={14} className="spin" /> : "Test"}
        </button>

        <div className="provider-test-result">
          {testResult && (
            <span className={testResult.success ? "success" : "error"}>
              {testResult.success ? <CheckCircle size={14} /> : <XCircle size={14} />}
              {testResult.message}
            </span>
          )}
        </div>

        <div className="provider-actions-right">
          <button
            onClick={() => onEdit(provider)}
            className="provider-action-btn icon edit"
            title="Edit"
          >
            <Edit2 size={16} />
          </button>
          <button
            onClick={() => deleteMutation.mutate()}
            disabled={deleteMutation.isPending}
            className="provider-action-btn icon delete"
            title="Delete"
          >
            {deleteMutation.isPending ? <Loader2 size={16} className="spin" /> : <Trash2 size={16} />}
          </button>
        </div>
      </div>
    </div>
  );
}
