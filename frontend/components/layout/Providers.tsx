// frontend/components/layout/Providers.tsx
"use client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type ReactNode, createContext, useContext, useEffect } from "react";

interface DevModeContextType {
  isDevMode: boolean;
  toggleDevMode: () => void;
}

export const DeveloperModeContext = createContext<DevModeContextType>({
  isDevMode: false,
  toggleDevMode: () => {},
});

export const useDeveloperMode = () => useContext(DeveloperModeContext);

export default function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(() => new QueryClient({
    defaultOptions: { queries: { retry: 1, staleTime: 30_000 } },
  }));

  const [isDevMode, setIsDevMode] = useState(false);
  
  useEffect(() => {
    const saved = localStorage.getItem("lecturemind_dev_mode");
    if (saved) setIsDevMode(saved === "true");
  }, []);

  const toggleDevMode = () => {
    setIsDevMode((prev) => {
      const next = !prev;
      localStorage.setItem("lecturemind_dev_mode", String(next));
      return next;
    });
  };

  return (
    <DeveloperModeContext.Provider value={{ isDevMode, toggleDevMode }}>
      <QueryClientProvider client={client}>
        {children}
      </QueryClientProvider>
    </DeveloperModeContext.Provider>
  );
}
