import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const skills = [
  {
    category: "Security",
    items: [
      "Penetration Testing",
      "Malware Analysis",
      "Network Security",
      "Incident Response"
    ]
  },
  {
    category: "Development",
    items: [
      "Python",
      "JavaScript",
      "TypeScript",
      "Java"
    ]
  },
  {
    category: "Cloud",
    items: [
      "Google Cloud",
      "Cloud Security",
      "Infrastructure as Code",
      "DevOps"
    ]
  },
  {
    category: "AI & ML",
    items: [
      "Machine Learning",
      "Neural Networks",
      "JAX",
      "TensorFlow"
    ]
  }
];

export function SkillsSection() {
  return (
    <section className="py-16 bg-gradient-to-b from-accent/5 to-background">
      <div className="container max-w-6xl">
        <h2 className="text-3xl font-bold mb-8 bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent">
          Skills & Expertise
        </h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {skills.map((skillGroup, index) => (
            <Card
              key={index}
              className={cn(
                "p-6 transition-all duration-300",
                "hover:shadow-lg hover:shadow-primary/10",
                "hover:scale-[1.02] dark:bg-card/80 backdrop-blur-sm",
                "border-primary/20 hover:border-primary/40",
                "bg-gradient-to-br from-card via-card/95 to-card/90"
              )}
            >
              <h3 className="font-semibold text-primary mb-4">
                {skillGroup.category}
              </h3>
              <ul className="space-y-2">
                {skillGroup.items.map((skill, skillIndex) => (
                  <li
                    key={skillIndex}
                    className="text-muted-foreground hover:text-foreground transition-colors flex items-center gap-2"
                  >
                    <div className="w-1.5 h-1.5 rounded-full bg-primary/60" />
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
