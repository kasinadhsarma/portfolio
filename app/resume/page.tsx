import { ResumeDropdown } from "@/components/ui/resume-dropdown";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { client } from "@/sanity/lib/client";
import { SanityResume } from "@/types/sanity";

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
    <div className="container mx-auto py-8">
      <h1 className="text-4xl font-bold mb-8">Resume</h1>

      <div className="mb-8">
        <ResumeDropdown />
      </div>

      <div className="grid gap-8">
        {/* Education Section */}
        {resume?.education && resume.education.length > 0 && (
          <section>
            <h2 className="text-2xl font-semibold mb-4">Education</h2>
            <div className="space-y-4">
              {resume.education.map((edu) => (
                <Card key={edu.institution} className="p-4">
                  <h3 className="text-xl font-semibold">{edu.institution}</h3>
                  <p className="text-muted-foreground">{edu.period}</p>
                  {edu.description && (
                    <p className="mt-2 whitespace-pre-line">{edu.description}</p>
                  )}
                </Card>
              ))}
            </div>
          </section>
        )}

        {/* Experience Section */}
        {resume?.experience && resume.experience.length > 0 && (
          <section>
            <h2 className="text-2xl font-semibold mb-4">Professional Experience</h2>
            <div className="space-y-4">
              {resume.experience.map((exp) => (
                <Card key={`${exp.title}-${exp.organization}`} className="p-4">
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="text-xl font-semibold">{exp.title}</h3>
                      <p className="text-muted-foreground">
                        {exp.organization} • {exp.period}
                      </p>
                    </div>
                    {exp.current && <Badge>Current</Badge>}
                  </div>
                  {exp.highlights && exp.highlights.length > 0 && (
                    <ul className="mt-4 list-disc pl-4 space-y-2">
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
            <h2 className="text-2xl font-semibold mb-4">Projects</h2>
            <div className="space-y-4">
              {resume.projects.map((project) => (
                <Card key={project.title} className="p-4">
                  <h3 className="text-xl font-semibold">{project.title}</h3>
                  {project.subtitle && (
                    <p className="text-muted-foreground">{project.subtitle}</p>
                  )}
                  {project.highlights && project.highlights.length > 0 && (
                    <ul className="mt-4 list-disc pl-4 space-y-2">
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
            <h2 className="text-2xl font-semibold mb-4">Training & Internships</h2>
            <div className="space-y-4">
              {resume.training.map((training) => (
                <Card key={`${training.title}-${training.organization}`} className="p-4">
                  <h3 className="text-xl font-semibold">{training.title}</h3>
                  <p className="text-muted-foreground">
                    {training.organization} • {training.period}
                  </p>
                  {training.description && (
                    <p className="mt-2">{training.description}</p>
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
