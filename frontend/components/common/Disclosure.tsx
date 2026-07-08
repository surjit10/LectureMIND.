// Reusable collapsible panel — progressive disclosure throughout the app.
"use client";
import { useState } from "react";

interface Props {
  label: string;
  icon?: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

export default function Disclosure({ label, icon = "▼", children, defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="disclosure">
      <button
        className="disclosure-trigger"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span>{icon} {label}</span>
        <span className="chevron">▼</span>
      </button>
      <div className="disclosure-body" hidden={!open}>
        {children}
      </div>
    </div>
  );
}
