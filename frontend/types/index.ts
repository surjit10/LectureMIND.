// frontend/types/index.ts
// TypeScript types matching the frozen backend schemas exactly.

export interface LectureInfo {
  lecture_id: string;
  title: string;
  display_name?: string;
  status: string;
  chunk_count: number;
  segment_count: number;
  duration?: number;
  course_name?: string;
  speaker?: string;
  description?: string;
  language?: string;
  created_at?: string;
}

export interface LectureRenameRequest {
  display_name: string;
}

export interface LectureRenameResponse {
  lecture_id: string;
  display_name: string;
  status: string;
}

export interface QueryRequest {
  query: string;
}

export interface EdgeProvenanceItem {
  source: string;
  target: string;
  confidence?: number;
  chunk_id?: string;
  timestamp?: number;
}

export interface PrerequisiteItem {
  concept: string;
  entity_id?: string;
  type?: string;
  depth: number;
  timestamp: number;
  chunk_id?: string;
  confidence: number;
  parent_concept?: string;
  target_concept?: string;
  path?: string[];
  edge_provenance?: EdgeProvenanceItem[];
  slide_text?: string;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  graph_path: string[];
  prerequisites?: PrerequisiteItem[];
  debug?: QueryDebug;
}

// Pipeline trace surfaced by the backend in Developer Mode.
export interface QueryDebug {
  retrieval_route: string;
  plan?: {
    intent?: string;
    is_lecture_wide?: boolean;
    answer_style?: string;
    top_k?: number;
    context_budget?: number;
    need_visual?: boolean;
  };
  telemetry?: Record<string, number>;
  stage_counts?: { vector: number; graph: number; reranked: number };
  vector_results?: { chunk_id: string; score: number }[];
  graph_results?: { start: string; related: string; rel_types: string[]; hops: number }[];
  reranked_results?: { chunk_id: string; rerank_score: number }[];
  final_context_chars?: number;
  graph_path?: string[];
  prerequisites?: PrerequisiteItem[];
}

export interface Source {
  chunk_id: string;
  timestamp: string;
  segment_id: string;
}

export interface StatusResponse {
  lecture_id: string;
  status: string;
  progress: number;
}

export interface UploadResponse {
  lecture_id: string;
  status: string;
  message: string;
}

export interface NotesResponse {
  lecture_id: string;
  notes: string;
}

export interface FlashcardsResponse {
  lecture_id: string;
  flashcards: { question: string; answer: string }[];
}

export interface QuizResponse {
  lecture_id: string;
  questions: { question: string; options: string[]; correct: string }[];
}

export interface LearningPathResponse {
  lecture_id: string;
  path: { title: string; description: string }[];
}

export interface SettingsUpdate {
  ollama_model?: string;
  top_k?: number;
}

export interface SettingsResponse {
  status: string;
  settings: Record<string, unknown>;
}

export type InferenceMode = "offline" | "online";

export type ProviderType =
  | "google_gemini"
  | "openai"
  | "groq"
  | "openrouter"
  | "anthropic"
  | "custom";

export interface Provider {
  id: string;
  provider: ProviderType;
  model: string;
  api_key: string;   // always "***" from API
  display_name: string;
  base_url: string | null;
}

export interface AISettings {
  inference_mode: InferenceMode;
  local_model: string;
  active_provider_id: string | null;
  providers: Provider[];
}

export interface AISettingsPatch {
  inference_mode?: InferenceMode;
  local_model?: string;
  active_provider_id?: string | null;
}

export interface ProviderAddRequest {
  provider: ProviderType;
  model: string;
  api_key: string;
  display_name?: string;
  base_url?: string;
}

export interface ProviderUpdateRequest {
  model?: string;
  api_key?: string;
  display_name?: string;
  base_url?: string;
}

export interface TestConnectionRequest {
  provider: ProviderType;
  api_key: string;
  base_url?: string;
}

export interface TestConnectionResult {
  success: boolean;
  message: string;
  models: string[];
}

export type PipelineStatus =
  | "UPLOADED"
  | "PROCESSING"
  | "SEGMENTING"
  | "EXTRACTING_ENTITIES"
  | "BUILDING_GRAPH"
  | "GENERATING_EMBEDDINGS"
  | "TRAINING_RERANKER"
  | "PACKAGING"
  | "READY"
  | "FAILED";
