"use client";

import { ScrollSection } from "@/components/ui/scroll-section";
import { HeroSection } from "@/components/pages/home/sections/hero-section";
import { AboutSection } from "@/components/pages/home/sections/about-section";
import { WhatImDoingSection } from "@/components/pages/home/sections/what-im-doing";
import { SkillsSection } from "@/components/pages/home/sections/skills-section";
import { CertificatesSection } from "@/components/pages/home/sections/certificates-section";
import { ContactCtaSection } from "@/components/pages/home/sections/contact-cta-section";
import { homePagePatterns as p } from "@/lib/responsive/pattrens/home";

export default function HomePage() {
  return (
    <div className={p.root}>
      {/* Background Effects */}
      <div className={p.backgroundWrapper}>
        <div className={p.backgroundGradient}></div>
      </div>

      <div className={p.contentWrapper}>
        <ScrollSection direction="down">
          <HeroSection />
        </ScrollSection>

        <ScrollSection direction="up">
          <AboutSection />
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
          <ContactCtaSection />
        </ScrollSection>
      </div>
    </div>
  );
}
