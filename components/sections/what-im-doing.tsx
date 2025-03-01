import { Card } from "@/components/ui/card";
import Image from "next/image";
import { cn } from "@/lib/utils";

const activities = [
  {
    icon: "/img/icon-design.svg",
    title: "Web Development",
    description: "Full-stack development with Python, JavaScript, and modern frameworks."
  },
  {
    icon: "/img/icon-dev.svg",
    title: "Cybersecurity Research",
    description: "Advanced security testing, vulnerability assessment, and malware analysis."
  },
  {
    icon: "/img/icons8-malware-94.png",
    title: "AI Development",
    description: "Building next-gen AI systems including SPU development and neural architectures."
  },
  {
    icon: "/img/google-cloud-icon-2048x1646-7admxejz.png",
    title: "Cloud Architecture",
    description: "Designing and implementing secure cloud solutions using Google Cloud Platform."
  }
];

export function WhatImDoingSection() {
  return (
    <section className="py-16 bg-gradient-to-b from-background to-accent/5">
      <div className="container max-w-6xl">
        <h2 className="text-3xl font-bold mb-8 bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent">
          What I&apos;m Doing
        </h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {activities.map((activity, index) => (
            <Card
              key={index}
              className={cn(
                "p-6 group transition-all duration-300",
                "hover:shadow-lg hover:shadow-primary/10",
                "hover:scale-[1.02] dark:bg-card/80 backdrop-blur-sm",
                "border-primary/20 hover:border-primary/40",
                "bg-gradient-to-br from-card via-card/95 to-card/90"
              )}
            >
              <div className="flex flex-col items-center text-center space-y-4">
                <div className={cn(
                  "w-16 h-16 rounded-xl bg-primary/10 flex items-center justify-center",
                  "group-hover:scale-110 transition-transform duration-300",
                  "group-hover:bg-primary/20"
                )}>
                  <Image
                    src={activity.icon}
                    alt={activity.title}
                    width={32}
                    height={32}
                    className="group-hover:scale-110 transition-transform duration-300"
                  />
                </div>
                <div className="space-y-2">
                  <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors">
                    {activity.title}
                  </h3>
                  <p className="text-sm text-muted-foreground group-hover:text-muted-foreground/80 line-clamp-2">
                    {activity.description}
                  </p>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
