"use client";

import Image from "next/image";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Mail, Phone, Github, Linkedin, X } from "lucide-react";
import { storeAndEncodeUrl, safeOpenUrl } from "@/lib/utils";
import { TypingEffect } from "@/components/ui/typing-effect";
import { ResumeDropdown } from "@/components/ui/resume-dropdown";
import { heroPatterns as p } from "@/lib/responsive/pattrens/home";

export function HeroSection() {
  return (
    <section className={p.section}>
      <div className={p.container}>
        <div className={p.layout}>
          <div className={p.avatarWrapper}>
            <div className={p.avatarFrame}>
              <div className={p.avatarGlow}></div>
              <Image
                src="/img/me_background.png"
                alt="Kasinadh Sarma"
                fill
                className={p.avatarImage}
                priority
              />
            </div>
          </div>
          <div className={p.contentColumn}>
            <h1 className={p.heading}>
              Kasinadh Sarma
            </h1>
            <div className={p.typingWrapper}>
              <TypingEffect
                texts={[
                  "Cybersecurity Engineer",
                  "Penetration Testing Engineer",
                  "DevSecOps Engineer",
                  "Full Stack Engineer",
                ]}
              />
            </div>
            <p className={p.tagline}>
              Bridging the gap between application security and full-stack engineering to build software that's both usable and hard to break.
            </p>
            <div className={p.ctaRow}>
              <Link href="mailto:kasinadhsarma@gmail.com">
                <Button
                  variant="outline"
                  className={p.ctaButton}
                >
                  <Mail className="w-4 h-4" />
                  Email Me
                </Button>
              </Link>
              <Link href="tel:+916305953487">
                <Button
                  variant="outline"
                  className={p.ctaButton}
                >
                  <Phone className="w-4 h-4" />
                  Call Me
                </Button>
              </Link>
              <ResumeDropdown />
            </div>
            <div className={p.socialRow}>
              <button
                aria-label="Visit my GitHub profile"
                onClick={() => safeOpenUrl(storeAndEncodeUrl("https://github.com/kasinadhsarma"))}
                className={p.socialLink}
              >
                <Github className="h-5 w-5" />
              </button>
              <button
                aria-label="Visit my LinkedIn profile"
                onClick={() => safeOpenUrl(storeAndEncodeUrl("https://www.linkedin.com/in/kasinadhsarma"))}
                className={p.socialLink}
              >
                <Linkedin className="h-5 w-5" />
              </button>
              <button
                aria-label="Visit my Twitter profile"
                onClick={() => safeOpenUrl(storeAndEncodeUrl("https://x.com/kasinadhsarma"))}
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
