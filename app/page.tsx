"use client";

import * as React from "react";
import Image from "next/image";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Mail, Phone, ExternalLink, Github, Linkedin, X } from "lucide-react";
import { storeAndEncodeUrl, safeOpenUrl } from "@/lib/utils";
import { useScrollAnimation } from "@/hooks/use-scroll-animation";
import { SkillsSection } from "@/components/sections/skills-section";
import { WhatImDoingSection } from "@/components/sections/what-im-doing";
import { cn } from "@/lib/utils";
import { ScrollSection } from "@/components/ui/scroll-section";
import { CertificatesSection } from "@/components/sections/certificates-section";
import { ResumeDropdown } from "@/components/ui/resume-dropdown";

// Typing Effect Component
function TypingEffect({ texts }: { texts: string[] }) {
  const [currentTextIndex, setCurrentTextIndex] = React.useState(0);
  const [currentText, setCurrentText] = React.useState("");
  const [isDeleting, setIsDeleting] = React.useState(false);

  React.useEffect(() => {
    const timeout = setTimeout(() => {
      const fullText = texts[currentTextIndex];
      if (!isDeleting) {
        setCurrentText(fullText.substring(0, currentText.length + 1));
        if (currentText === fullText) {
          setIsDeleting(true);
          setTimeout(() => {}, 1000);
        }
      } else {
        setCurrentText(fullText.substring(0, currentText.length - 1));
        if (currentText === "") {
          setIsDeleting(false);
          setCurrentTextIndex((currentTextIndex + 1) % texts.length);
        }
      }
    }, isDeleting ? 50 : 100);
    return () => clearTimeout(timeout);
  }, [currentText, currentTextIndex, isDeleting, texts]);

  return (
    <span className="text-xl font-medium bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent dark:from-primary/90 dark:via-primary/70 dark:to-primary/50">
      {currentText}
      <span className="animate-blink text-primary">|</span>
    </span>
  );
}

// Animated Section Component
function AnimatedSection({ children, className }: { children: React.ReactNode; className?: string }) {
  const { ref, isVisible } = useScrollAnimation();
  return (
    <section
      ref={ref}
      className={cn(
        "transform transition-all duration-700 ease-out",
        isVisible ? "translate-y-0 opacity-100" : "translate-y-8 opacity-0",
        className
      )}
    >
      {children}
    </section>
  );
}

export default function HomePage() {
  return (
    <div className="relative">
      {/* Background Effects */}
      <div className="fixed inset-0 z-0">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-accent/5 via-background to-background"></div>
      </div>

      <div className="relative z-10">
        <ScrollSection direction="down">
          {/* Hero Section */}
          <section className="relative py-20">
            <div className="container max-w-6xl">
              <div className="flex flex-col md:flex-row items-center gap-12">
                <div className="flex-shrink-0">
                  <div className="relative w-48 h-48 md:w-64 md:h-64">
                    <div className="absolute inset-0 bg-gradient-to-r from-accent/20 to-accent/30 rounded-full blur-2xl opacity-50 animate-pulse"></div>
                    <Image
                      src="/img/me.jpg"
                      alt="Kasinadh Sarma"
                      fill
                      className="rounded-full object-cover border-4 border-accent/20"
                      priority
                    />
                  </div>
                </div>
                <div className="flex-1 text-center md:text-left">
                  <h1 className="text-4xl md:text-5xl font-bold mb-6 text-foreground">
                    Kasinadh Sarma
                  </h1>
                  <div className="h-10 mb-8">
                  <TypingEffect
                      texts={[
                        "Pentration Tester",
                        "AI Researcher",
                        "Machine Learning Researcher",
                        "Cybersecurity Researcher",
                        "Quantum Researcher",
                        "Neural Networking and Cloud AI Researcher & Developer",
                        "Full Stack Engineer",
                      ]}
                    />

                </div>
                <p className="text-muted-foreground mb-8 max-w-2xl">
                    Bridging the gap between cybersecurity and innovation through cutting-edge research and development.
                  </p>
                  <div className="flex flex-wrap gap-4 justify-center md:justify-start">
                    <Link href="mailto:kasinadhsarma@gmail.com">
                      <Button 
                        variant="outline" 
                        className="gap-2"
                      >
                        <Mail className="w-4 h-4" />
                        Email Me
                      </Button>
                    </Link>
                    <Link href="tel:+916305953487">
                      <Button 
                        variant="outline" 
                        className="gap-2"
                      >
                        <Phone className="w-4 h-4" />
                        Call Me
                      </Button>
                    </Link>
                    <ResumeDropdown />
                  </div>
                  <div className="flex gap-4 mt-8 justify-center md:justify-start">
                    <button
                      aria-label="Visit my GitHub profile"
                      onClick={() => safeOpenUrl(storeAndEncodeUrl("https://github.com/kasinadhsarma"))}
                      className="text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <Github className="h-5 w-5" />
                    </button>
                    <button
                      aria-label="Visit my LinkedIn profile"
                      onClick={() => safeOpenUrl(storeAndEncodeUrl("https://www.linkedin.com/in/kasinadhsarma"))}
                      className="text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <Linkedin className="h-5 w-5" />
                    </button>
                    <button
                      aria-label="Visit my Twitter profile"
                      onClick={() => safeOpenUrl(storeAndEncodeUrl("https://x.com/kasinadhsarma"))}
                      className="text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <X className="h-5 w-5" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </ScrollSection>

        <ScrollSection direction="up">
          {/* About Section */}
          <AnimatedSection className="py-16">
            <div className="container max-w-6xl">
              <h2 className="text-3xl font-bold mb-8 bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent">
                About Me
              </h2>
              <div className="prose dark:prose-invert max-w-none">
                <p className="mb-4 text-foreground dark:text-foreground/90">
                  I'm a cybersecurity enthusiast with a B.Tech in Cyber/Computer Forensics and Counterterrorism from Parul University. My mission is to enhance the cybersecurity and AI landscapes through innovative technologies and solutions.
                </p>
                <p className="mb-4 text-foreground dark:text-foreground/90">
                  I'm Kasinadh Sarma, an AI and cybersecurity researcher with expertise in building secure, production-grade web applications. With a CSE background and hands-on experience in full-stack development (Next.js, TypeScript, React), I specialize in creating scalable solutions that prioritize security-first architecture. I've implemented enterprise-level security automation and maintained zero-vulnerability production systems in healthcare technology                </p>
              </div>
            </div>
          </AnimatedSection>
        </ScrollSection>

        <ScrollSection direction="up">
          <WhatImDoingSection />
        </ScrollSection>

        <ScrollSection direction="left">
          <SkillsSection />
        </ScrollSection>

        <ScrollSection direction="right">
          <CertificatesSection />
        </ScrollSection>

        <ScrollSection direction="up" delay={200}>
          {/* Contact CTA */}
          <AnimatedSection className="py-16">
            <div className="container max-w-6xl">
              <div className="relative p-12 rounded-3xl overflow-hidden text-center bg-gradient-to-br from-card via-card/95 to-card/90 dark:from-card/90 dark:to-card/70 border border-primary/20 hover:border-primary/40 shadow-lg hover:shadow-primary/10">
                <div className="relative z-10 max-w-3xl mx-auto">
                  <h2 className="text-3xl font-bold mb-6 bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent">
                    Let's Work Together
                  </h2>
                  <p className="text-muted-foreground dark:text-muted-foreground/80 mb-8">
                    I'm available for freelance work, collaborations, and research opportunities.
                    Let's connect and build something amazing together.
                  </p>
                  <Link href="mailto:kasinadhsarma@gmail.com">
                    <Button 
                      variant="default" 
                      size="lg" 
                      className="px-8 bg-primary hover:bg-primary/90 text-primary-foreground transition-all duration-300 hover:scale-105"
                    >
                      Get in Touch
                    </Button>
                  </Link>
                </div>
                <div className="absolute inset-0 bg-gradient-to-r from-primary/10 via-primary/5 to-primary/10 dark:from-primary/5 dark:via-primary/2 dark:to-primary/5 opacity-50"></div>
              </div>
            </div>
          </AnimatedSection>
        </ScrollSection>
      </div>
    </div>
  );
}
