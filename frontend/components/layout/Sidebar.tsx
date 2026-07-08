// frontend/components/layout/Sidebar.tsx
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useDeveloperMode } from "./Providers";

import { LayoutDashboard, UploadCloud, Settings } from "lucide-react";

const NAV = [
  { href: "/", label: "Dashboard", icon: <LayoutDashboard size={20} /> },
  { href: "/upload", label: "Upload", icon: <UploadCloud size={20} /> },
  { href: "/settings", label: "Settings", icon: <Settings size={20} /> },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { isDevMode, toggleDevMode } = useDeveloperMode();
  
  return (
    <aside className="sidebar flex flex-col justify-between h-full">
      <div>
        <div className="sidebar-brand">
          <h2>LectureMind</h2>
          <span className="sidebar-version">V7</span>
        </div>
        <nav className="sidebar-nav">
          {NAV.map((n) => (
            <Link key={n.href} href={n.href}
              className={`nav-link ${pathname === n.href ? "active" : ""}`}>
              <span className="nav-icon">{n.icon}</span>{n.label}
            </Link>
          ))}
        </nav>
      </div>
      
      <div className="p-4 mt-auto border-t border-white/10">
        <div className="flex items-center justify-between text-sm text-gray-400">
          <span>Dev Mode</span>
          <button 
            onClick={toggleDevMode}
            className={`px-2 py-1 rounded text-xs font-mono transition-colors ${
              isDevMode ? 'bg-primary text-white' : 'bg-gray-800 text-gray-500'
            }`}
          >
            {isDevMode ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>
    </aside>
  );
}
