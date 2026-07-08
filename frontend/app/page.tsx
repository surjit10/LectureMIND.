// Dashboard — Screen 1
"use client";
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getLectures, renameLecture, deleteLecture } from "@/services/api";
import StatusBadge from "@/components/common/StatusBadge";
import ErrorMessage from "@/components/common/ErrorMessage";
import Link from "next/link";
import { useDeveloperMode } from "@/components/layout/Providers";
import { Clock, Layers, FileText, FolderOpen, PackageOpen, Edit2, Trash2, Check, Activity, Search, AlertTriangle, ArrowRight } from "lucide-react";
import { formatDistanceToNow } from "date-fns";

export default function Dashboard() {
  const queryClient = useQueryClient();
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["lectures"],
    queryFn: getLectures,
  });
  
  const { isDevMode } = useDeveloperMode();

  const [renameModal, setRenameModal] = useState<{ id: string, oldName: string } | null>(null);
  const [deleteModal, setDeleteModal] = useState<{ id: string, name: string } | null>(null);
  const [newName, setNewName] = useState("");
  const [searchQuery, setSearchQuery] = useState("");

  const renameMutation = useMutation({
    mutationFn: (vars: { id: string, name: string }) => renameLecture(vars.id, vars.name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["lectures"] });
      setRenameModal(null);
    }
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteLecture(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["lectures"] });
      setDeleteModal(null);
    }
  });

  const filteredData = data?.filter(l => {
    const title = l.display_name || l.title || "Untitled Lecture";
    return title.toLowerCase().includes(searchQuery.toLowerCase());
  });

  return (
    <div className="dashboard-shell">
      {/* Header */}
      <div className="dashboard-header">
        <div className="dashboard-header-title">
          <h1 className="dashboard-title">Knowledge Library</h1>
          <p className="dashboard-subtitle">Manage your imported packages</p>
        </div>
        <div className="dashboard-header-actions">
          <div className="dashboard-search">
            <Search size={16} />
            <input 
              type="text" 
              placeholder="Search library..." 
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
            />
          </div>
          <Link href="/upload" className="btn-primary">
            <UploadIcon /> Import Package
          </Link>
        </div>
      </div>
      
      {isLoading && (
        <div className="dashboard-grid">
          {[1, 2, 3].map(i => (
            <div key={i} className="dashboard-card-skeleton animate-pulse"></div>
          ))}
        </div>
      )}
      
      {error && <ErrorMessage message={error.message} onRetry={() => refetch()} />}
      
      {data && data.length === 0 && (
        <div className="dashboard-empty-state">
          <div className="empty-icon-box">
            <PackageOpen size={48} />
          </div>
          <h3>Your library is empty</h3>
          <p>Import your first knowledge package to start learning interactively.</p>
          <Link href="/upload" className="btn-primary mt-4">Import Knowledge</Link>
        </div>
      )}
      
      {filteredData && filteredData.length > 0 && (
        <div className="dashboard-grid">
          {filteredData.map((l) => {
            const title = l.display_name || l.title || "Untitled Lecture";
            return (
              <div key={l.lecture_id} className="dashboard-card group">
                {/* Glow Effect */}
                <div className="dashboard-card-glow" />

                <div className="dashboard-card-content">
                  <div className="flex justify-between items-start mb-4">
                    <StatusBadge status={l.status} />
                    <button 
                      onClick={() => setDeleteModal({ id: l.lecture_id, name: title })}
                      className="dashboard-card-action hover:text-red-400"
                      title="Delete"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                  
                  <Link href={`/lecture/${l.lecture_id}`} className="dashboard-card-title-link">
                    <h3 className="line-clamp-2">{title}</h3>
                  </Link>
                  
                  <div className="dashboard-card-meta">
                    <div className="meta-item"><Layers size={14} /> {l.chunk_count} Chunks</div>
                    <div className="meta-item"><FileText size={14} /> {l.segment_count} Segments</div>
                    {l.duration !== undefined && l.duration > 0 && (
                      <div className="meta-item"><Clock size={14} /> {Math.round(l.duration / 60)}m</div>
                    )}
                  </div>
                  
                  <div className="dashboard-card-footer">
                    <div className="footer-date">
                      <Activity size={14} /> 
                      {l.created_at ? formatDistanceToNow(new Date(l.created_at)) + ' ago' : 'Unknown'}
                    </div>
                    <button 
                      onClick={() => { setRenameModal({ id: l.lecture_id, oldName: title }); setNewName(title); }}
                      className="dashboard-card-action hover:text-primary"
                      title="Rename"
                    >
                      <Edit2 size={14} />
                    </button>
                  </div>

                  {isDevMode && (
                    <div className="dev-mode-badge mt-3">ID: {l.lecture_id}</div>
                  )}
                </div>

                <Link href={`/lecture/${l.lecture_id}`} className="dashboard-card-overlay-btn">
                  Open <ArrowRight size={16} />
                </Link>
              </div>
            );
          })}
        </div>
      )}

      {/* Rename Modal */}
      {renameModal && (
        <div className="settings-modal-overlay">
          <div className="settings-modal">
            <div className="settings-modal-header">
              <h2 className="settings-modal-title">Rename Package</h2>
              <button onClick={() => setRenameModal(null)} className="settings-modal-close"><X size={20} /></button>
            </div>
            <div className="settings-form">
              <div className="settings-form-group">
                <label className="settings-label">Old Name</label>
                <div className="settings-input opacity-50 bg-transparent">{renameModal.oldName}</div>
              </div>
              <div className="settings-form-group">
                <label className="settings-label">New Name</label>
                <input 
                  type="text" 
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  className="settings-input"
                  autoFocus
                />
              </div>
            </div>
            <div className="settings-modal-actions mt-8">
              <button onClick={() => setRenameModal(null)} className="btn-secondary">Cancel</button>
              <button 
                onClick={() => renameMutation.mutate({ id: renameModal.id, name: newName })}
                disabled={newName.trim().length < 3 || renameMutation.isPending}
                className="btn-primary"
              >
                {renameMutation.isPending ? "Saving..." : "Save Changes"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Modal */}
      {deleteModal && (
        <div className="settings-modal-overlay">
          <div className="settings-modal border-red-500/30">
            <div className="settings-modal-icon">
              <AlertTriangle size={28} />
            </div>
            <h2 className="settings-modal-title">Delete Package?</h2>
            <p className="text-white font-medium mb-2">{deleteModal.name}</p>
            <p className="settings-modal-desc">
              This cannot be undone. All metadata, indexes, and stored files will be permanently removed.
            </p>
            <div className="settings-modal-actions">
              <button onClick={() => setDeleteModal(null)} className="btn-secondary">Cancel</button>
              <button 
                onClick={() => deleteMutation.mutate(deleteModal.id)}
                disabled={deleteMutation.isPending}
                className="btn-danger"
              >
                {deleteMutation.isPending ? "Deleting..." : "Delete Permanently"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function UploadIcon() {
  return <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>;
}

function X({ size }: { size: number }) {
  return <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>;
}
