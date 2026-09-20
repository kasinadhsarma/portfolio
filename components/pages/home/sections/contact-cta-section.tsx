"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { AnimatedSection } from "@/components/ui/animated-section";
import { contactCtaPatterns as p } from "@/lib/responsive/pattrens/home";

export function ContactCtaSection() {
  return (
    <AnimatedSection className={p.section}>
      <div className={p.container}>
        <div className={p.card}>
          <div className={p.content}>
            <h2 className={p.heading}>
              Let's Work Together
            </h2>
            <p className={p.description}>
              I'm available for freelance work, collaborations, and research opportunities.
              Let's connect and build something amazing together.
            </p>
            <Link href="mailto:kasinadhsarma@gmail.com">
              <Button
                variant="default"
                size="lg"
                className={p.button}
              >
                Get in Touch
              </Button>
            </Link>
          </div>
          <div className={p.glow}></div>
        </div>
      </div>
    </AnimatedSection>
  );
}
