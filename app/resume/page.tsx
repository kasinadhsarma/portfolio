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
                  <p className="text-muted-foreground">Ropods Spot • June 2025 — Present</p>
                </div>
                <Badge>Current</Badge>
              </div>
              <ul className="mt-4 list-disc pl-4 space-y-2">
                <li>Contributing to the company's technology stack modernization and digital transformation initiatives</li>
                <li>Collaborating with cross-functional teams to deliver high-quality software solutions</li>
              </ul>
            </Card>

            <Card className="p-4">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="text-xl font-semibold">Research and Development Lead</h3>
                  <p className="text-muted-foreground">VishwamAI • 2023 — Present</p>
                </div>
                <Badge>Current</Badge>
              </div>
              <ul className="mt-4 list-disc pl-4 space-y-2">
                <li>vishwamai is under research and devlopment mode</li>
              </ul>
              <a
                href="https://github.com/VishwamAI"
                className="text-primary hover:underline mt-2 inline-block"
                target="_blank"
                rel="noopener noreferrer"
              >
                View GitHub Repository →
              </a>
            </Card>
          </div>
        </section>

        {/* Practical Experience Section */}
        <section>
          <h2 className="text-2xl font-semibold mb-4">Practical Experience</h2>
          <div className="space-y-4">
            <Card className="p-4">
              <h3 className="text-xl font-semibold">Summer Trainee</h3>
              <p className="text-muted-foreground">Intel • May 2023 — July 2023</p>
              <p className="mt-2">
                Participated in the Intel Unnati summer training program. Learned about various machine learning projects and their implementation in different modes.
              </p>
            </Card>

            <Card className="p-4">
              <h3 className="text-xl font-semibold">Cybersecurity Trainee</h3>
              <p className="text-muted-foreground">Try Hack Me • September 2022 — February 2023</p>
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
      </div>
    </div>
  );
}
