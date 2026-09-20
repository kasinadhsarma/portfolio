"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { useScrollAnimation } from "@/hooks/use-scroll-animation";
import { animatedSectionPatterns as p } from "@/lib/responsive/pattrens/ui";

export function AnimatedSection({ children, className }: { children: React.ReactNode; className?: string }) {
  const { ref, isVisible } = useScrollAnimation();
  return (
    <section
      ref={ref}
      className={cn(
        p.base,
        isVisible ? p.visible : p.hidden,
        className
      )}
    >
      {children}
    </section>
  );
}
