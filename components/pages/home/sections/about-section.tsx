"use client";

import { AnimatedSection } from "@/components/ui/animated-section";
import { aboutPatterns as p } from "@/lib/responsive/pattrens/home";

export function AboutSection() {
  return (
    <AnimatedSection className={p.section}>
      <div className={p.container}>
        <h2 className={p.heading}>
          About Me
        </h2>
        <div className={p.proseWrapper}>
          <p className={p.paragraph}>
            I'm a Software Engineer with a B.Tech in Cyber Security from Parul University, specializing in Application Security (AppSec), DevSecOps, and full-stack web development.
          </p>
          <p className={p.paragraph}>
            I'm Kasinadh Sarma. I build secure, production-grade web applications with Next.js, TypeScript, and Node.js, integrating security into the SDLC by hardening API routes, enforcing strict secrets hygiene in CI/CD pipelines, and maintaining zero-vulnerability dependencies. I also work as a penetration tester, identifying and helping remediate vulnerabilities across web applications and infrastructure.
          </p>
        </div>
      </div>
    </AnimatedSection>
  );
}
