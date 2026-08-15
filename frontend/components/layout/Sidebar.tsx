// frontend/components/layout/Sidebar.tsx
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useDeveloperMode } from "./Providers";
import { LayoutDashboard, UploadCloud, Settings, Sparkles, Terminal } from "lucide-react";

const NAV = [
  { href: "/", label: "Dashboard", icon: <LayoutDashboard size={19} /> },
  { href: "/upload", label: "Import Package", icon: <UploadCloud size={19} /> },
  { href: "/settings", label: "Settings", icon: <Settings size={19} /> },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { isDevMode, toggleDevMode } = useDeveloperMode();
  
  return (
    <aside className="sidebar flex flex-col justify-between h-full">
      <div>
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">
            <Sparkles size={20} />
          </div>
          <h2>LectureMind</h2>
        </div>
        <nav className="sidebar-nav">
          {NAV.map((n) => {
            const isActive = pathname === n.href;
            return (
              <Link 
                key={n.href} 
                href={n.href}
                className={`nav-link ${isActive ? "active" : ""}`}
              >
                <span className="nav-icon">{n.icon}</span>
                <span className="nav-text">{n.label}</span>
                {isActive && <div className="nav-indicator" />}
              </Link>
            );
          })}
        </nav>
      </div>
      
      <div className="sidebar-footer">
        <div className="sidebar-dev-box">
          <div className="sidebar-dev-label">
            <Terminal size={14} className="text-gray-400" />
            <span>Developer Mode</span>
          </div>
          <button 
            onClick={toggleDevMode}
            className={`dev-toggle-btn ${isDevMode ? 'active' : ''}`}
            aria-label="Toggle developer mode"
          >
            <div className="dev-toggle-knob" />
          </button>
        </div>
      </div>
    </aside>
  );
}
