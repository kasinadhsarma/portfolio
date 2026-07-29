import { ResumeDropdown } from "@/components/ui/resume-dropdown";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function ResumePage() {
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-4xl font-bold mb-8">Resume</h1>

      <div className="mb-8">
        <ResumeDropdown />
      </div>

      <div className="grid gap-8">
        {/* Education Section */}
        <section>
          <h2 className="text-2xl font-semibold mb-4">Education</h2>
          <div className="space-y-4">
            <Card className="p-4">
              <h3 className="text-xl font-semibold">Parul University</h3>
              <p className="text-muted-foreground">2020 — 2024</p>
              <p className="mt-2">
                Researching various topics and integrating life and passion into one module.
                <br />
                Grade Point Average: 7.46
                <br />
                Results to be declared in May 2024, exams to be completed by March 2024 in Vadodara.
              </p>
            </Card>

            <Card className="p-4">
              <h3 className="text-xl font-semibold">Sri Chaitanya J.R College</h3>
              <p className="text-muted-foreground">2018 — 2020</p>
              <p className="mt-2">
                Completed intermediate with a percentage of 6.9 at Sri Chaitanya J.R. College in Guntur.
              </p>
            </Card>

            <Card className="p-4">
              <h3 className="text-xl font-semibold">Sri Chaitanya Techno School</h3>
              <p className="text-muted-foreground">2017 — 2018</p>
              <p className="mt-2">
                Completed 10th grade with a grade point average of 7.7 in Guntur.
              </p>
            </Card>
          </div>
        </section>

        {/* Experience Section */}
        <section>
          <h2 className="text-2xl font-semibold mb-4">Professional Experience</h2>
          <div className="space-y-4">
            <Card className="p-4">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="text-xl font-semibold">Junior Engineer</h3>
                  <p className="text-muted-foreground">ROPODS (Medtech Startup) • June 2025 — Present</p>
                </div>
                <Badge>Current</Badge>
              </div>
              <ul className="mt-4 list-disc pl-4 space-y-2">
                <li>Secured the shop.ropods.com storefront and Razorpay payment integration; engineered internal API routes with rigorous input validation and RBAC checks, securing 100% of checkout flows</li>
                <li>Engineered a dual-layer CAPTCHA system (Google reCAPTCHA v3 + custom logic challenge) across public-facing endpoints to thwart bot attacks and form spam</li>
                <li>Architected secure Next.js deployment pipelines on Vercel via GitHub Actions, enforcing strict environment-scoped secrets management</li>
                <li>Drove continuous SAST and dependency auditing, maintaining zero known high-severity CVEs and resolving 100% of security-related TypeScript/ESLint errors</li>
                <li>Conducted research on UCP (Universal Commerce Protocol) and ACP (Agentic Commerce Protocol) for product-to-platform communication standards</li>
              </ul>
            </Card>
          </div>
        </section>

        {/* Projects Section */}
        <section>
          <h2 className="text-2xl font-semibold mb-4">Projects</h2>
          <div className="space-y-4">
            <Card className="p-4">
              <h3 className="text-xl font-semibold">CIELON</h3>
              <p className="text-muted-foreground">Freelance Client Project</p>
              <ul className="mt-4 list-disc pl-4 space-y-2">
                <li>Architected the secure backend infrastructure for a greenfield Next.js 16 storefront, enforcing strict TypeScript type-safety across all server-side functions</li>
                <li>Built a secure content synchronization pipeline between Sanity Studio and Firestore with a custom read cache and strict Firebase Security Rules</li>
                <li>Authored backend security architecture documentation and database schema designs focused on data confidentiality and Zero Trust principles</li>
              </ul>
            </Card>
          </div>
        </section>

        {/* Practical Experience Section */}
        <section>
          <h2 className="text-2xl font-semibold mb-4">Training & Internships</h2>
          <div className="space-y-4">
            <Card className="p-4">
              <h3 className="text-xl font-semibold">Summer Trainee</h3>
              <p className="text-muted-foreground">Intel Unnati • May 2023 — July 2023</p>
              <p className="mt-2">
                Built an AI-powered road object detection system using OpenCV, implementing memory-safe image preprocessing and strict input validation to mitigate buffer overflows and adversarial input vectors. Designed secure data ingestion architectures for large-scale TensorFlow datasets, ensuring data integrity and tamper-proofing throughout the training lifecycle.
              </p>
            </Card>

            <Card className="p-4">
              <h3 className="text-xl font-semibold">Cybersecurity Trainee</h3>
              <p className="text-muted-foreground">TryHackMe • September 2022 — February 2023</p>
              <p className="mt-2">
                Enrolled on TryHackMe to learn and practice cybersecurity skills, participated in challenges and completed learning paths to enhance knowledge.
              </p>
            </Card>

            <Card className="p-4">
              <h3 className="text-xl font-semibold">Cybersecurity Practice Trainee</h3>
              <p className="text-muted-foreground">PicoCTF • November 2022 — April 2023</p>
              <p className="mt-2">
                Actively engaged with the platform to enhance cybersecurity skills through practical challenges and learning paths.
              </p>
            </Card>
          </div>
        </section>

        {/* Certifications Section */}
        <section>
          <h2 className="text-2xl font-semibold mb-4">Certifications</h2>
          <div className="grid gap-3 md:grid-cols-2">
            {[
              { title: "Pre Security (SEC0) Certification", issuer: "TryHackMe", url: "https://www.credly.com/badges/4c08443d-1a8f-425c-a1bd-c466d23e8b87/public_url" },
              { title: "Google Cloud Skills Boost", issuer: "Google Cloud" },
              { title: "Google Professional Cyber Security", issuer: "Credly" },
              { title: "Cybersecurity Roles, Processes & Operating System Security", issuer: "Coursera, IBM" },
              { title: "Introduction of IoT", issuer: "NPTEL" },
              { title: "Introduction to Programming Using Python", issuer: "HackerRank" },
              { title: "Networking Basics", issuer: "Cisco" },
              { title: "Networking Devices and Initial Configuration", issuer: "Cisco" },
            ].map((cert) => (
              <Card key={cert.title} className="p-4">
                {cert.url ? (
                  <a href={cert.url} target="_blank" rel="noopener noreferrer" className="font-semibold hover:text-primary transition-colors">
                    {cert.title}
                  </a>
                ) : (
                  <h3 className="font-semibold">{cert.title}</h3>
                )}
                <p className="text-sm text-muted-foreground">{cert.issuer}</p>
              </Card>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
