// components/settings/OnlineConfig.tsx
// Config panel shown when Inference Mode = "online".
// Renders the list of configured providers and allows adding new ones.

"use client";
import { useState } from "react";
import type { Provider, AISettings } from "@/types";
import ProviderCard from "./ProviderCard";
import ProviderDialog from "./ProviderDialog";
import { Plus } from "lucide-react";

interface Props {
  settings: AISettings;
}

export default function OnlineConfig({ settings }: Props) {
  const [showDialog, setShowDialog] = useState(false);
  const [editTarget, setEditTarget] = useState<Provider | null>(null);

  const handleEdit = (p: Provider) => {
    setEditTarget(p);
    setShowDialog(true);
  };

  const handleAdd = () => {
    setEditTarget(null);
    setShowDialog(true);
  };

  return (
    <div className="provider-config-container">
      {/* Provider list */}
      {settings.providers.length === 0 ? (
        <div className="provider-empty-state">
          <p>No online providers configured yet.</p>
          <button id="add-first-provider" onClick={handleAdd} className="btn-primary">
            <Plus size={16} /> Add First Provider
          </button>
        </div>
      ) : (
        <div className="provider-cards-list">
          {settings.providers.map((p) => (
            <ProviderCard
              key={p.id}
              provider={p}
              isActive={p.id === settings.active_provider_id}
              onEdit={handleEdit}
            />
          ))}

          <button
            id="add-provider-btn"
            onClick={handleAdd}
            className="provider-add-card"
          >
            <div className="provider-add-icon">
              <Plus size={24} />
            </div>
            <span>Add New Provider</span>
          </button>
        </div>
      )}

      {/* Add/edit dialog */}
      {showDialog && (
        <ProviderDialog
          editTarget={editTarget}
          onClose={() => { setShowDialog(false); setEditTarget(null); }}
        />
      )}
    </div>
  );
}
