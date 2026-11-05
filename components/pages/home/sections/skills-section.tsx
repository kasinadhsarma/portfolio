import { Card } from "@/components/ui/card";
import { skillsPatterns as p } from "@/lib/responsive/pattrens/home";
import type { SkillGroup } from "@/types/content";

interface SkillsSectionProps {
  skills: SkillGroup[];
}

export function SkillsSection({ skills }: SkillsSectionProps) {
  return (
    <section className={p.section}>
      <div className={p.container}>
        <h2 className={p.heading}>
          Skills &amp; Expertise
        </h2>
        <div className={p.grid}>
          {skills.map((skillGroup, index) => (
            <Card key={index} className={p.card}>
              <h3 className={p.categoryTitle}>
                {skillGroup.category}
              </h3>
              <ul className={p.list}>
                {skillGroup.items.map((skill, skillIndex) => (
                  <li key={skillIndex} className={p.listItem}>
                    <div className={p.bullet} />
                    {skill}
                  </li>
                ))}
              </ul>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
