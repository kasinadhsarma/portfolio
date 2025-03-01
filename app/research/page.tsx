import { cn } from "@/lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { PublicationCard } from "@/components/publication-card"
import { ResearchCard } from "@/components/research-card"

export default function ResearchPage() {
  const publications = [
    {
      title: "Developing Optimized Large Language Models on Limited Compute Resources",
      url: "https://easychair.org/publications/preprint/gP6v",
    },
    {
      title: "Chip Technology and Development: Advancements in Nanotechnology and Beyond",
      url: "https://easychair.org/publications/preprint/t3bK",
    },
    {
      title: "Revolutionizing Machine Learning: the Emergence and Impact of Google’s TPU Technology",
      url: "https://easychair.org/publications/preprint/KFMC",
    },
    {
      title: "Exploring AI Consciousness: Integrating Theory, Practice, and the NeuroFlex Architecture",
      url: "https://easychair.org/publications/preprint/FlGf",
    },
    {
      title: "Sustainable Oxygen Production and Plant Cultivation Strategies for Future Earth Scenarios",
      url: "https://easychair.org/publications/preprint/wdDt",
    },
    {
      title: "EVM.ova Security Assessment: a Penetration Testing and Vulnerability Analysis Project",
      url: "https://easychair.org/publications/preprint/jRJp",
    },
  ]

  const papers = [
    {
      title: "Accelerating Quantum Computing Through AI Co-Design",
      path: "/pdf/Accelerating Quantum Computing Through AI Co-Desig.pdf",
    },
    {
      title: "Architectural Framework and Implementation Strategy",
      path: "/pdf/Architectural Framework and Implementation Strateg.pdf",
    },
    {
      title: "Comprehensive Development Strategy for Next-Generation Computing",
      path: "/pdf/Comprehensive Development Strategy for Next-Genera.pdf",
    },
    {
      title: "Reasoning Based Quantum Computing Acceleration",
      path: "/pdf/do reasioning based quntum faster and superstic.pdf",
    },
    {
      title: "Neuromorphic Computing Architectures: Biological Inspiration",
      path: "/pdf/Neuromorphic Computing Architectures_ Biological I.pdf",
    },
    {
      title: "Quantum Computing at Room Temperature: Methodology",
      path: "/pdf/Quantum Computing at Room Temperature_ Methodologi.pdf",
    },
    {
      title: "Room-Temperature Quantum Computing: Engineering Considerations",
      path: "/pdf/Room-Temperature Quantum Computing_ Engineering Co.pdf",
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
        <h2 className="text-2xl font-semibold mb-6">Research Papers</h2>
        <div className="grid gap-6">
          {papers.map((paper) => (
            <PublicationCard
              key={paper.path}
              title={paper.title}
              url={paper.path}
              isLocal
            />
          ))}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-6">Ongoing Research</h2>
        <div className="grid gap-6">
          <ResearchCard
            title="AI Consciousness and Cognitive Architectures"
            status="Ongoing"
            description="Research focusing on developing computational models of consciousness using advanced AI frameworks and cognitive science theories."
            technologies={["JAX", "Flax", "Optax", "PyTorch"]}
          />
          <ResearchCard
            title="Hardware Acceleration for Neural Networks"
            status="Active"
            description="Investigating and developing novel hardware acceleration techniques for neural network training and inference."
            technologies={["CUDA", "TPU", "FPGA", "Custom Hardware"]}
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
