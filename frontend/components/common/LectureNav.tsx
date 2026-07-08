"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  MessageSquare,
  FileText,
  Layers,
  HelpCircle,
  Map,
} from "lucide-react";

interface NavTab {
  href: string;
  label: string;
  icon: React.ReactNode;
  color: string; // CSS color for the active accent
}

interface LectureNavProps {
  lectureId: string;
  /** Some pages use segmentId-based routes. Pass it to override. */
  segmentId?: string;
}

export default function LectureNav({ lectureId, segmentId }: LectureNavProps) {
  const pathname = usePathname();

  const id = segmentId ?? lectureId;

  const tabs: NavTab[] = [
    {
      href: `/chat/${lectureId}`,
      label: "Chat",
      icon: <MessageSquare size={17} />,
      color: "#8b5cf6",
    },
    {
      href: `/notes/${id}`,
      label: "Notes",
      icon: <FileText size={17} />,
      color: "#3b82f6",
    },
    {
      href: `/flashcards/${id}`,
      label: "Flashcards",
      icon: <Layers size={17} />,
      color: "#10b981",
    },
    {
      href: `/quiz/${id}`,
      label: "Quiz",
      icon: <HelpCircle size={17} />,
      color: "#f59e0b",
    },
    {
      href: `/learning-path/${lectureId}`,
      label: "Learning Path",
      icon: <Map size={17} />,
      color: "#ec4899",
    },
  ];

  return (
    <nav className="lecture-nav-bar">
      {tabs.map((tab) => {
        const isActive =
          pathname === tab.href ||
          pathname.startsWith(tab.href + "/") ||
          // fuzzy: matches /chat/..., /notes/..., etc.
          pathname.startsWith("/" + tab.href.split("/")[1]);

        return (
          <Link
            key={tab.href}
            href={tab.href}
            className={`lecture-nav-tab ${isActive ? "active" : ""}`}
            style={
              isActive
                ? ({
                    "--tab-color": tab.color,
                    "--tab-bg": tab.color + "20",
                    "--tab-border": tab.color + "50",
                  } as React.CSSProperties)
                : undefined
            }
          >
            <span className="lecture-nav-icon">{tab.icon}</span>
            <span>{tab.label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
