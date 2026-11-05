"use client";

import Image from "next/image";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Mail, Phone, Github, Linkedin, X } from "lucide-react";
import { storeAndEncodeUrl, safeOpenUrl } from "@/lib/utils";
import { TypingEffect } from "@/components/ui/typing-effect";
import { ResumeDropdown } from "@/components/ui/resume-dropdown";
import { heroPatterns as p } from "@/lib/responsive/pattrens/home";
import type { HeroContent, ResumeFileCard } from "@/types/content";

interface HeroSectionProps {
  hero: HeroContent;
  resumeFiles: ResumeFileCard[];
}

export function HeroSection({ hero, resumeFiles }: HeroSectionProps) {
  return (
    <section className={p.section}>
      <div className={p.container}>
        <div className={p.layout}>
          <div className={p.avatarWrapper}>
            <div className={p.avatarFrame}>
              <div className={p.avatarGlow}></div>
              <Image
                src={hero.avatarImage}
                alt={hero.name}
                fill
                className={p.avatarImage}
                priority
              />
            </div>
          </div>
          <div className={p.contentColumn}>
            <h1 className={p.heading}>
              {hero.name}
            </h1>
            <div className={p.typingWrapper}>
              <TypingEffect texts={hero.roles} />
            </div>
            <p className={p.tagline}>
              {hero.tagline}
            </p>
            <div className={p.ctaRow}>
              <Link href={`mailto:${hero.email}`}>
                <Button variant="outline" className={p.ctaButton}>
                  <Mail className="w-4 h-4" />
                  Email Me
                </Button>
              </Link>
              <Link href={`tel:${hero.phone}`}>
                <Button variant="outline" className={p.ctaButton}>
                  <Phone className="w-4 h-4" />
                  Call Me
                </Button>
              </Link>
              <ResumeDropdown resumeFiles={resumeFiles} />
            </div>
            <div className={p.socialRow}>
              <button
                aria-label="Visit my GitHub profile"
                onClick={() => safeOpenUrl(storeAndEncodeUrl(hero.github))}
                className={p.socialLink}
              >
                <Github className="h-5 w-5" />
              </button>
              <button
                aria-label="Visit my LinkedIn profile"
                onClick={() => safeOpenUrl(storeAndEncodeUrl(hero.linkedin))}
                className={p.socialLink}
              >
                <Linkedin className="h-5 w-5" />
              </button>
              <button
                aria-label="Visit my Twitter profile"
                onClick={() => safeOpenUrl(storeAndEncodeUrl(hero.twitter))}
                className={p.socialLink}
              >
                <X className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
