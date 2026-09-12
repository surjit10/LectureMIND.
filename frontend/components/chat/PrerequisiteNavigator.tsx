// frontend/components/chat/PrerequisiteNavigator.tsx
// Socratic Prerequisite Back-Tracker UI component.
//
// Renders conceptual dependency breadcrumbs and branching trees.
// Links each prerequisite to exact video timestamps, slide provenance,
// and confidence scores, labeled as Inferred Pedagogical Hypotheses.

"use client";

import React, { useState } from "react";
import { Compass, Clock, ArrowRight, ChevronDown, ChevronRight, Sparkles, Layers } from "lucide-react";
import { PrerequisiteItem } from "@/types";

interface PrerequisiteNavigatorProps {
  prerequisites: PrerequisiteItem[];
  onSeek?: (seconds: number) => void;
}

function formatTimestamp(seconds: number): string {
  if (isNaN(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s < 10 ? "0" : ""}${s}`;
}

export function PrerequisiteNavigator({
  prerequisites,
  onSeek,
}: PrerequisiteNavigatorProps) {
  const [expanded, setExpanded] = useState<boolean>(false);
  const [selectedItem, setSelectedItem] = useState<PrerequisiteItem | null>(null);

  if (!prerequisites || prerequisites.length === 0) {
    return null;
  }

  // Sort by depth DESC (deepest prerequisite first, leading to immediate parent)
  const sorted = [...prerequisites].sort((a, b) => {
    if (b.depth !== a.depth) return b.depth - a.depth;
    return a.timestamp - b.timestamp;
  });

  const handlePillClick = (item: PrerequisiteItem) => {
    if (onSeek && item.timestamp > 0) {
      onSeek(item.timestamp);
    }
    setSelectedItem(selectedItem?.concept === item.concept ? null : item);
  };

  return (
    <div className="prereq-navigator-card">
      <div
        className="prereq-header"
        onClick={() => setExpanded(!expanded)}
        title="Click to toggle prerequisite dependency details"
      >
        <div className="prereq-title-group">
          <div className="prereq-badge-icon">
            <Compass size={14} className="prereq-compass-icon" />
          </div>
          <span className="prereq-heading">Socratic Prerequisite Path</span>
          <span className="prereq-inferred-tag">Inferred</span>
        </div>
        <button
          type="button"
          className="prereq-toggle-btn"
          aria-label={expanded ? "Collapse prerequisites" : "Expand prerequisites"}
        >
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>
      </div>

      {/* Sequential Breadcrumb Bar */}
      <div className="prereq-chain-row">
        {sorted.map((item, idx) => {
          const isSelected = selectedItem?.concept === item.concept;
          const confPercent = Math.round((item.confidence || 0.8) * 100);

          return (
            <React.Fragment key={`${item.concept}-${item.depth}-${idx}`}>
              <button
                type="button"
                className={`prereq-pill ${isSelected ? "selected" : ""}`}
                onClick={() => handlePillClick(item)}
                title={`Prerequisite: ${item.concept} (Confidence: ${confPercent}%, Hop Depth: ${item.depth})`}
              >
                <span className="prereq-step-num">Step {sorted.length - item.depth + 1}</span>
                <span className="prereq-name">{item.concept}</span>
                {item.timestamp > 0 && (
                  <span className="prereq-time">
                    <Clock size={11} />
                    {formatTimestamp(item.timestamp)}
                  </span>
                )}
                <span className="prereq-conf-badge">{confPercent}%</span>
              </button>

              {idx < sorted.length - 1 && (
                <div className="prereq-arrow">
                  <ArrowRight size={12} />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Expanded Provenance & Branching Tree Details */}
      {expanded && (
        <div className="prereq-details-pane">
          <div className="prereq-pane-header">
            <Layers size={13} />
            <span>Dependency Provenance &amp; Multi-Hop Traversal</span>
          </div>

          <div className="prereq-list">
            {sorted.map((item, idx) => (
              <div
                key={`detail-${item.concept}-${idx}`}
                className={`prereq-detail-row ${selectedItem?.concept === item.concept ? "active" : ""}`}
              >
                <div className="prereq-detail-top">
                  <span className="prereq-concept-title">
                    {idx + 1}. {item.concept}
                  </span>
                  <div className="prereq-meta-badges">
                    <span className="prereq-depth-badge">Depth: {item.depth} hop{item.depth > 1 ? "s" : ""}</span>
                    <span className="prereq-conf-badge-pill">
                      <Sparkles size={10} />
                      {Math.round((item.confidence || 0.8) * 100)}% confidence
                    </span>
                  </div>
                </div>

                {item.path && item.path.length > 1 && (
                  <div className="prereq-path-subrow">
                    <span className="prereq-sublabel">Full Path:</span>
                    <span className="prereq-path-nodes">
                      {item.path.join(" ➔ ")}
                    </span>
                  </div>
                )}

                {item.edge_provenance && item.edge_provenance.length > 0 && (
                  <div className="prereq-edge-hops">
                    {item.edge_provenance.map((edge, eIdx) => (
                      <div key={eIdx} className="prereq-edge-hop-item">
                        <span className="prereq-hop-edge">
                          {edge.source} ➔ {edge.target}
                        </span>
                        {edge.timestamp !== undefined && edge.timestamp > 0 && (
                          <button
                            type="button"
                            className="prereq-jump-link"
                            onClick={() => onSeek && onSeek(edge.timestamp!)}
                          >
                            <Clock size={11} />
                            {formatTimestamp(edge.timestamp)}
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
