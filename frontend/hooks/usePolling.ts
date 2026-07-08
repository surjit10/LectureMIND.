// frontend/hooks/usePolling.ts
// Polling hook for status updates (5-second interval).

"use client";
import { useQuery } from "@tanstack/react-query";
import { getStatus } from "@/services/api";

export function useStatusPolling(lectureId: string, enabled = true) {
  return useQuery({
    queryKey: ["status", lectureId],
    queryFn: () => getStatus(lectureId),
    refetchInterval: enabled ? 5000 : false,
    enabled,
  });
}
