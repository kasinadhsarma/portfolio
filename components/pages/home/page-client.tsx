"use client";

import { ScrollSection } from "@/components/ui/scroll-section";
import { HeroSection } from "@/components/pages/home/sections/hero-section";
import { AboutSection } from "@/components/pages/home/sections/about-section";
import { WhatImDoingSection } from "@/components/pages/home/sections/what-im-doing";
import { SkillsSection } from "@/components/pages/home/sections/skills-section";
import { AchievementsSection } from "@/components/pages/home/sections/achievements-section";
import { CertificatesSection } from "@/components/pages/home/sections/certificates-section";
import { ContactCtaSection } from "@/components/pages/home/sections/contact-cta-section";
import { homePagePatterns as p } from "@/lib/responsive/pattrens/home";
import type {
  HeroContent,
  AboutContent,
  ActivityCard,
  SkillGroup,
  CertificateCard,
  ResumeFileCard,
  AchievementCard,
} from "@/types/content";

interface HomePageClientProps {
  hero: HeroContent;
  about: AboutContent;
  activities: ActivityCard[];
  skills: SkillGroup[];
  certificates: CertificateCard[];
  resumeFiles: ResumeFileCard[];
  achievements: AchievementCard[];
}

export default function HomePageClient({
  hero,
  about,
  activities,
  skills,
  certificates,
  resumeFiles,
  achievements,
}: HomePageClientProps) {
  return (
    <div className={p.root}>
      {/* Background Effects */}
      <div className={p.backgroundWrapper}>
        <div className={p.backgroundGradient}></div>
      </div>

      <div className={p.contentWrapper}>
        <ScrollSection direction="down">
          <HeroSection hero={hero} resumeFiles={resumeFiles} />
        </ScrollSection>

        <ScrollSection direction="up">
          <AboutSection about={about} />
        </ScrollSection>

        <ScrollSection direction="up">
          <WhatImDoingSection activities={activities} />
        </ScrollSection>

        <ScrollSection direction="left">
          <SkillsSection skills={skills} />
        </ScrollSection>

        {achievements.length > 0 && (
          <ScrollSection direction="up">
            <AchievementsSection achievements={achievements} />
          </ScrollSection>
        )}

        <ScrollSection direction="right">
          <CertificatesSection certificates={certificates} />
        </ScrollSection>

        <ScrollSection direction="up" delay={200}>
          <ContactCtaSection />
        </ScrollSection>
      </div>
    </div>
  );
}
