import { getProjects } from "@/lib/content"
import { ProjectsHeader } from "@/components/pages/projects/projects-header"
import { ProjectsClient } from "./projects-client"
import { projectsPagePatterns as p } from "@/lib/responsive/pattrens/projects"

export default async function ProjectsPage() {
  const projects = await getProjects()

  return (
    <div className={p.container}>
      <ProjectsHeader />
      <ProjectsClient projects={projects} />
    </div>
  )
}
