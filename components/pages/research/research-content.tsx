import { PublicationCard } from "@/components/cards/publication-card"
import { ResearchCard } from "@/components/cards/research-card"
import { researchContentPatterns as p } from "@/lib/responsive/pattrens/research"

const publications = [
  {
    title: "EVM.ova Security Assessment: a Penetration Testing and Vulnerability Analysis Project",
    url: "https://easychair.org/publications/preprint/jRJp",
  },
]

export function ResearchContent() {
  return (
    <div className={p.container}>
      <div>
        <h1 className={p.title}>
          Research Work
        </h1>
        <div className={p.divider}></div>
      </div>

      <section>
        <h2 className={p.sectionHeading}>Published Research</h2>
        <div className={p.grid}>
          {publications.map((pub) => (
            <PublicationCard
              key={pub.url}
              title={pub.title}
              url={pub.url}
            />
          ))}
        </div>
      </section>

      <section>
        <h2 className={p.sectionHeading}>Ongoing Research</h2>
        <div className={p.grid}>
          <ResearchCard
            title="DevSecOps Automation for CI/CD Pipelines"
            status="Ongoing"
            description="Researching secrets management, SAST, and dependency-auditing automation to keep production deployments free of high-severity CVEs."
            technologies={["GitHub Actions", "Vercel", "SAST", "Dependency Auditing"]}
          />
          <ResearchCard
            title="Web Application Penetration Testing Methodologies"
            status="Active"
            description="Investigating structured approaches to vulnerability analysis, injection testing, and API security assessment on production web applications."
            technologies={["Burp Suite", "Nmap", "Metasploit", "OWASP Top 10"]}
          />
          <ResearchCard
            title="Cybersecurity in Cloud Computing"
            status="Ongoing"
            description="Research on advanced security measures and threat detection in cloud computing environments."
            technologies={["Google Cloud", "AWS", "Azure", "Security Tools"]}
          />
        </div>
      </section>
    </div>
  )
}
