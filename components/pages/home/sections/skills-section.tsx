import { Card } from "@/components/ui/card";
import { skillsPatterns as p } from "@/lib/responsive/pattrens/home";

const skills = [
  {
    category: "Security & AppSec",
    items: [
      "Penetration Testing",
      "API Hardening",
      "Input Validation",
      "Secrets Hygiene",
      "Dependency Auditing",
      "RBAC"
    ]
  },
  {
    category: "Development",
    items: [
      "Next.js",
      "React",
      "Node.js",
      "TypeScript",
      "Python",
      "REST APIs"
    ]
  },
  {
    category: "Cloud & DevSecOps",
    items: [
      "Google Cloud Platform",
      "Firebase Security Rules",
      "Vercel",
      "GitHub Actions CI/CD Security"
    ]
  },
  {
    category: "Languages & Tools",
    items: [
      "Python",
      "TypeScript",
      "Dart",
      "Bash",
      "Git",
      "Linux"
    ]
  }
];

export function SkillsSection() {
  return (
    <section className={p.section}>
      <div className={p.container}>
        <h2 className={p.heading}>
          Skills & Expertise
        </h2>
        <div className={p.grid}>
          {skills.map((skillGroup, index) => (
            <Card
              key={index}
              className={p.card}
            >
              <h3 className={p.categoryTitle}>
                {skillGroup.category}
              </h3>
              <ul className={p.list}>
                {skillGroup.items.map((skill, skillIndex) => (
                  <li
                    key={skillIndex}
                    className={p.listItem}
                  >
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
