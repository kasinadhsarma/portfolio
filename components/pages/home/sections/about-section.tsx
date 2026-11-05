"use client";

import { AnimatedSection } from "@/components/ui/animated-section";
import { aboutPatterns as p } from "@/lib/responsive/pattrens/home";
import type { AboutContent } from "@/types/content";

interface AboutSectionProps {
  about: AboutContent;
}

export function AboutSection({ about }: AboutSectionProps) {
  return (
    <AnimatedSection className={p.section}>
      <div className={p.container}>
        <h2 className={p.heading}>
          About Me
        </h2>
        <div className={p.proseWrapper}>
          {about.paragraphs.map((text, i) => (
            <p key={i} className={p.paragraph}>
              {text}
            </p>
          ))}
        </div>
      </div>
    </AnimatedSection>
  );
}
