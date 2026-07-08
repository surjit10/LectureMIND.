// components/settings/InferenceModeSelector.tsx
// Renders the Offline / Online mode toggle buttons.

"use client";
import type { InferenceMode } from "@/types";
import { Monitor, Cloud, CheckCircle2 } from "lucide-react";

interface Props {
  value: InferenceMode;
  onChange: (mode: InferenceMode) => void;
  disabled?: boolean;
}

export default function InferenceModeSelector({ value, onChange, disabled }: Props) {
  return (
    <div className="mode-selector-container">
      <label className="settings-section-label">Select Processing Environment</label>
      <div className="mode-cards-grid">
        <ModeCard
          id="mode-offline"
          active={value === "offline"}
          icon={<Monitor size={32} />}
          label="Local Processing"
          description="Runs LLM models directly on your hardware via Ollama. Maximum privacy, zero latency, no internet required."
          onClick={() => onChange("offline")}
          disabled={disabled}
        />
        <ModeCard
          id="mode-online"
          active={value === "online"}
          icon={<Cloud size={32} />}
          label="Cloud Intelligence"
          description="Connects to powerful remote APIs like Gemini, OpenAI, or Groq for the highest quality responses."
          onClick={() => onChange("online")}
          disabled={disabled}
        />
      </div>
    </div>
  );
}

function ModeCard({
  id, active, icon, label, description, onClick, disabled,
}: {
  id: string;
  active: boolean;
  icon: React.ReactNode;
  label: string;
  description: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      id={id}
      onClick={onClick}
      disabled={disabled}
      className={`mode-card ${active ? "active" : ""}`}
    >
      <div className="mode-card-bg"></div>
      
      {/* Checkmark icon (Top right) */}
      <div className={`mode-card-check ${active ? "visible" : ""}`}>
        <CheckCircle2 size={24} />
      </div>

      <div className="mode-card-content">
        <div className="mode-card-icon">{icon}</div>
        <h3 className="mode-card-title">{label}</h3>
        <p className="mode-card-desc">{description}</p>
      </div>
    </button>
  );
}
