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
import { SanityResumeFileCard } from "@/types/sanity";
import Link from "next/link";

interface ResumeDropdownProps {
  variant?: "default" | "outline";
  size?: "default" | "sm" | "lg" | "icon";
  className?: string;
}

export function ResumeDropdown({
  variant = "default",
  size = "default",
  className
}: ResumeDropdownProps) {
  const [resumes, setResumes] = useState<SanityResumeFileCard[]>([]);

  useEffect(() => {
    fetch("/api/resume-files")
      .then((res) => res.json())
      .then((data) => setResumes(Array.isArray(data) ? data : []))
      .catch((error) => console.error("Failed to load resume files:", error));
  }, []);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant={variant} size={size} className={cn("gap-2", className)}>
          <FileDown className="h-4 w-4" />
          Download Resume
          <ChevronDown className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        {resumes.map((resume) => (
          <DropdownMenuItem key={resume.url} asChild>
            <Link
              href={resume.url}
              download
              className="flex items-center gap-2 cursor-pointer w-full"
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
