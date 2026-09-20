import { ResumeDropdown } from "@/components/ui/resume-dropdown";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { client } from "@/sanity/lib/client";
import { SanityResume } from "@/types/sanity";
import { resumePagePatterns as p } from "@/lib/responsive/pattrens/resume";

async function getResume(): Promise<SanityResume | null> {
  try {
    return await client.fetch<SanityResume>(`
      *[_type == "resume"][0] {
        education,
        experience,
        projects,
        training
      }
    `)
  } catch (error) {
    console.error('Failed to fetch resume from Sanity:', error)
    return null
  }
}

export default async function ResumePage() {
  const resume = await getResume();

  return (
    <div className={p.container}>
      <h1 className={p.title}>Resume</h1>

      <div className={p.dropdownWrapper}>
        <ResumeDropdown />
      </div>

      <div className={p.sectionsGrid}>
        {/* Education Section */}
        {resume?.education && resume.education.length > 0 && (
          <section>
            <h2 className={p.sectionHeading}>Education</h2>
            <div className={p.itemsWrapper}>
              {resume.education.map((edu) => (
                <Card key={edu.institution} className={p.card}>
                  <h3 className={p.itemTitle}>{edu.institution}</h3>
                  <p className={p.itemMeta}>{edu.period}</p>
                  {edu.description && (
                    <p className={p.descriptionPreLine}>{edu.description}</p>
                  )}
                </Card>
              ))}
            </div>
          </section>
        )}

        {/* Experience Section */}
        {resume?.experience && resume.experience.length > 0 && (
          <section>
            <h2 className={p.sectionHeading}>Professional Experience</h2>
            <div className={p.itemsWrapper}>
              {resume.experience.map((exp) => (
                <Card key={`${exp.title}-${exp.organization}`} className={p.card}>
                  <div className={p.expHeaderRow}>
                    <div>
                      <h3 className={p.itemTitle}>{exp.title}</h3>
                      <p className={p.itemMeta}>
                        {exp.organization} • {exp.period}
                      </p>
                    </div>
                    {exp.current && <Badge>Current</Badge>}
                  </div>
                  {exp.highlights && exp.highlights.length > 0 && (
                    <ul className={p.highlightsList}>
                      {exp.highlights.map((highlight) => (
                        <li key={highlight}>{highlight}</li>
                      ))}
                    </ul>
                  )}
                </Card>
              ))}
            </div>
          </section>
        )}

        {/* Projects Section */}
        {resume?.projects && resume.projects.length > 0 && (
          <section>
            <h2 className={p.sectionHeading}>Projects</h2>
            <div className={p.itemsWrapper}>
              {resume.projects.map((project) => (
                <Card key={project.title} className={p.card}>
                  <h3 className={p.itemTitle}>{project.title}</h3>
                  {project.subtitle && (
                    <p className={p.itemMeta}>{project.subtitle}</p>
                  )}
                  {project.highlights && project.highlights.length > 0 && (
                    <ul className={p.highlightsList}>
                      {project.highlights.map((highlight) => (
                        <li key={highlight}>{highlight}</li>
                      ))}
                    </ul>
                  )}
                </Card>
              ))}
            </div>
          </section>
        )}

        {/* Practical Experience Section */}
        {resume?.training && resume.training.length > 0 && (
          <section>
            <h2 className={p.sectionHeading}>Training & Internships</h2>
            <div className={p.itemsWrapper}>
              {resume.training.map((training) => (
                <Card key={`${training.title}-${training.organization}`} className={p.card}>
                  <h3 className={p.itemTitle}>{training.title}</h3>
                  <p className={p.itemMeta}>
                    {training.organization} • {training.period}
                  </p>
                  {training.description && (
                    <p className={p.descriptionPlain}>{training.description}</p>
                  )}
                </Card>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
