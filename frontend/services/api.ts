// frontend/services/api.ts
// Single source of truth for all API calls.
// No direct fetch calls inside components — everything goes through this file.

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  try {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json", ...options?.headers },
      ...options,
    });
    if (!res.ok) {
      const errorBody = await res.text().catch(() => "Unknown error");
      throw new Error(`API ${res.status}: ${errorBody}`);
    }
    return res.json();
  } catch (err) {
    if (err instanceof TypeError && err.message.includes("fetch")) {
      throw new Error("Backend unavailable. Please check your connection.");
    }
    throw err;
  }
}

// GET /lectures
export const getLectures = () =>
  request<import("@/types").LectureInfo[]>("/lectures");

// GET /lecture/{id}
export const getLecture = (id: string) =>
  request<import("@/types").LectureInfo>(`/lecture/${id}`);

// POST /upload — import a LectureMind Knowledge Package (.zip)
export const uploadLecture = async (file: File, displayName?: string, description?: string) => {
  const formData = new FormData();
  // Do NOT set Content-Type manually — browser sets multipart boundary automatically.
  formData.append("file", file);
  if (displayName) formData.append("display_name", displayName.trim());
  if (description) formData.append("description", description.trim());

  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
  if (!res.ok) {
    // Always try to extract the backend `detail` field for a user-friendly message.
    const text = await res.text().catch(() => "");
    let detail = text;
    try {
      const json = JSON.parse(text);
      if (json?.detail) detail = String(json.detail);
    } catch { /* keep raw text */ }
    throw new Error(detail || `Import failed with status ${res.status}.`);
  }
  return res.json() as Promise<import("@/types").UploadResponse>;
};

// GET /status/{id}
export const getStatus = (id: string) =>
  request<import("@/types").StatusResponse>(`/status/${id}`);

// POST /query
export const postQuery = (lectureId: string, query: string) =>
  request<import("@/types").QueryResponse>("/query", {
    method: "POST",
    body: JSON.stringify({ query, lecture_id: lectureId }),
  });

// POST /notes
export const postNotes = (lectureId: string) =>
  request<import("@/types").NotesResponse>("/notes", {
    method: "POST",
    body: JSON.stringify({ lecture_id: lectureId }),
  });

// POST /flashcards
export const postFlashcards = (lectureId: string, count = 10) =>
  request<import("@/types").FlashcardsResponse>("/flashcards", {
    method: "POST",
    body: JSON.stringify({ lecture_id: lectureId, count }),
  });

// POST /quiz
export const postQuiz = (lectureId: string, count = 5) =>
  request<import("@/types").QuizResponse>("/quiz", {
    method: "POST",
    body: JSON.stringify({ lecture_id: lectureId, count }),
  });

// POST /learning_path
export const postLearningPath = (lectureId: string) =>
  request<import("@/types").LearningPathResponse>("/learning_path", {
    method: "POST",
    body: JSON.stringify({ lecture_id: lectureId }),
  });

// PUT /settings (legacy — kept for backward compat)
export const updateSettings = (settings: import("@/types").SettingsUpdate) =>
  request<import("@/types").SettingsResponse>("/settings", {
    method: "PUT",
    body: JSON.stringify(settings),
  });

// GET /settings
export const getSettings = () =>
  request<import("@/types").AISettings>("/settings");

// PATCH /settings
export const patchSettings = (patch: Partial<import("@/types").AISettingsPatch>) =>
  request<import("@/types").AISettings>("/settings", {
    method: "PATCH",
    body: JSON.stringify(patch),
  });

// GET /settings/providers
export const getProviders = () =>
  request<import("@/types").Provider[]>("/settings/providers");

// POST /settings/providers
export const addProvider = (body: import("@/types").ProviderAddRequest) =>
  request<import("@/types").Provider>("/settings/providers", {
    method: "POST",
    body: JSON.stringify(body),
  });

// PATCH /settings/providers/{id}
export const updateProvider = (id: string, body: import("@/types").ProviderUpdateRequest) =>
  request<import("@/types").Provider>(`/settings/providers/${id}`, {
    method: "PATCH",
    body: JSON.stringify(body),
  });

// DELETE /settings/providers/{id}
export const deleteProvider = async (id: string) => {
  const url = `${API_BASE}/settings/providers/${id}`;
  const res = await fetch(url, { method: "DELETE" });
  if (!res.ok) {
    const errorBody = await res.text().catch(() => "Unknown error");
    throw new Error(`API ${res.status}: ${errorBody}`);
  }
};

// POST /settings/providers/{id}/test
export const testProvider = (id: string) =>
  request<import("@/types").TestConnectionResult>(`/settings/providers/${id}/test`, {
    method: "POST",
  });

// POST /settings/providers/test
export const testNewProvider = (body: import("@/types").TestConnectionRequest) =>
  request<import("@/types").TestConnectionResult>("/settings/providers/test", {
    method: "POST",
    body: JSON.stringify(body),
  });

// GET /settings/ollama/models
export const getOllamaModels = () =>
  request<{ models: string[] }>("/settings/ollama/models");


// PATCH /lectures/{id}/rename
export const renameLecture = (lectureId: string, displayName: string) =>
  request<import("@/types").LectureRenameResponse>(`/lectures/${lectureId}/rename`, {
    method: "PATCH",
    body: JSON.stringify({ display_name: displayName }),
  });

// DELETE /lectures/{id}
export const deleteLecture = async (lectureId: string) => {
  const url = `${API_BASE}/lectures/${lectureId}`;
  const res = await fetch(url, { method: "DELETE" });
  if (!res.ok) {
    const errorBody = await res.text().catch(() => "Unknown error");
    throw new Error(`API ${res.status}: ${errorBody}`);
  }
};

// DELETE /lectures
export const resetAllLectures = async () => {
  const url = `${API_BASE}/lectures`;
  const res = await fetch(url, { method: "DELETE" });
  if (!res.ok) {
    const errorBody = await res.text().catch(() => "Unknown error");
    throw new Error(`API ${res.status}: ${errorBody}`);
  }
};

// --- Global Reranker API ---

export interface RerankerStatusResponse {
  status: string;
  model_name: string;
  custom_model: boolean;
  loaded: boolean;
  disk_size_mb: number;
  last_updated: string;
  max_upload_size_mb: number;
  backend: string;
}

// GET /api/reranker/status
export const getGlobalRerankerStatus = () =>
  request<RerankerStatusResponse>("/api/reranker/status");

// POST /api/reranker/upload
export const uploadGlobalReranker = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);
  
  const res = await fetch(`${API_BASE}/api/reranker/upload`, { method: "POST", body: formData });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    let detail = text;
    try {
      const json = JSON.parse(text);
      if (json?.detail) detail = String(json.detail);
    } catch { /* keep raw text */ }
    throw new Error(detail || `Upload failed with status ${res.status}.`);
  }
  return res.json();
};

// POST /api/reranker/reload
export const reloadGlobalReranker = () =>
  request<{status: string, message: string}>("/api/reranker/reload", { method: "POST" });

// DELETE /api/reranker
export const deleteGlobalReranker = async () => {
  const url = `${API_BASE}/api/reranker/`;
  const res = await fetch(url, { method: "DELETE" });
  if (!res.ok) {
    const errorBody = await res.text().catch(() => "Unknown error");
    throw new Error(`API ${res.status}: ${errorBody}`);
  }
};

// PATCH /api/reranker/settings
export const patchGlobalRerankerSettings = (max_upload_size_mb: number) =>
  request<{ status: string; max_upload_size_mb: number }>("/api/reranker/settings", {
    method: "PATCH",
    body: JSON.stringify({ max_upload_size_mb }),
  });

