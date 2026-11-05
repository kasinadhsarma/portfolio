'use client'

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import { ChevronDown, FileDown } from "lucide-react";
import { ResumeFileCard } from "@/types/content";
import { resumeDropdownPatterns as p } from "@/lib/responsive/pattrens/ui";
import Link from "next/link";

interface ResumeDropdownProps {
  variant?: "default" | "outline";
  size?: "default" | "sm" | "lg" | "icon";
  className?: string;
  /** Pre-fetched resume files. When provided, skips the client-side fetch. */
  resumeFiles?: ResumeFileCard[];
}

export function ResumeDropdown({
  variant = "default",
  size = "default",
  className,
  resumeFiles: initialFiles,
}: ResumeDropdownProps) {
  const [resumes, setResumes] = useState<ResumeFileCard[]>(initialFiles ?? []);

  useEffect(() => {
    // Only fetch if no pre-fetched data was supplied
    if (initialFiles && initialFiles.length > 0) return;
    fetch("/api/resume-files")
      .then((res) => res.json())
      .then((data) => setResumes(Array.isArray(data) ? data : []))
      .catch((error) => console.error("Failed to load resume files:", error));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant={variant} size={size} className={cn(p.triggerButton, className)}>
          <FileDown className="h-4 w-4" />
          Download Resume
          <ChevronDown className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className={p.content}>
        {resumes.map((resume) => (
          <DropdownMenuItem key={resume.url} asChild>
            <Link
              href={resume.url}
              download
              className={p.link}
            >
              <FileDown className="h-4 w-4" />
              <span>{resume.label}</span>
            </Link>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
