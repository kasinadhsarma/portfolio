import { cn } from "@/lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { PublicationCard } from "@/components/publication-card"
import { ResearchCard } from "@/components/research-card"

export default function ResearchPage() {
  const publications = [
    {
      title: "EVM.ova Security Assessment: a Penetration Testing and Vulnerability Analysis Project",
      url: "https://easychair.org/publications/preprint/jRJp",
    },
  ]

  return (
    <div className="container mx-auto px-4 py-12 space-y-12">
      <div>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-primary to-primary/50 bg-clip-text text-transparent">
          Research Work
        </h1>
        <div className="h-1 w-20 bg-gradient-to-r from-primary to-primary/50 mt-2"></div>
      </div>

      <section>
        <h2 className="text-2xl font-semibold mb-6">Published Research</h2>
        <div className="grid gap-6">
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
        <h2 className="text-2xl font-semibold mb-6">Ongoing Research</h2>
        <div className="grid gap-6">
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
            status="Completed"
            description="Research on advanced security measures and threat detection in cloud computing environments."
            technologies={["Google Cloud", "AWS", "Azure", "Security Tools"]}
          />
        </div>
      </section>
    </div>
  )
}
