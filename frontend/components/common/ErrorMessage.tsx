// frontend/components/common/ErrorMessage.tsx
// User-friendly error display — never exposes stack traces.
"use client";

interface Props { message: string; onRetry?: () => void }

export default function ErrorMessage({ message, onRetry }: Props) {
  // If the error message comes from our backend, it might already be user friendly 
  // (e.g. "Knowledge package validation failed. Missing required files: manifest.json")
  // We should preserve these exact messages, while masking unhandled network crashes.
  
  let friendly = message;
  
  if (message.includes("fetch") || message.includes("unavailable") && !message.includes("temporarily")) {
    friendly = "Backend unavailable. Please check your connection.";
  } else if (message.includes("Ollama") || message.includes("timeout")) {
    friendly = "Language model is not responding. Please try again later.";
  } else if (message.includes("Neo4j") && !message.includes("Import")) {
    friendly = "Knowledge graph service is temporarily unavailable.";
  } else if (message.includes("Qdrant") && !message.includes("Import")) {
    friendly = "Vector search service is temporarily unavailable.";
  } else if (message === "Unknown error") {
    friendly = "Something went wrong. Please try again.";
  }

  return (
    <div className="error-card bg-red-900/20 border border-red-500/30 p-4 rounded-xl flex flex-col items-start">
      <div className="flex items-start gap-3">
        <span className="text-red-500 mt-0.5">⚠️</span>
        <p className="error-text text-red-200 text-sm font-medium">{friendly}</p>
      </div>
      {onRetry && <button className="mt-3 ml-7 px-4 py-1.5 bg-red-500/20 hover:bg-red-500/30 text-red-300 rounded text-sm transition-colors" onClick={onRetry}>Retry</button>}
    </div>
  );
}
