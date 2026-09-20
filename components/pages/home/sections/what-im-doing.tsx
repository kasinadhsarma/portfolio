import { Card } from "@/components/ui/card";
import Image from "next/image";
import { whatImDoingPatterns as p } from "@/lib/responsive/pattrens/home";

const activities = [
  {
    icon: "/img/icon-design.svg",
    title: "Web Development",
    description: "Full-stack development with Next.js, React, Node.js, and TypeScript."
  },
  {
    icon: "/img/icon-dev.svg",
    title: "Cybersecurity Research",
    description: "Advanced security testing, vulnerability assessment, and malware analysis."
  },
  {
    icon: "/img/icons8-malware-94.png",
    title: "Penetration Testing",
    description: "Identifying and exploiting vulnerabilities in web apps and infrastructure to strengthen security posture."
  },
  {
    icon: "/img/google-cloud-icon-2048x1646-7admxejz.png",
    title: "Cloud Architecture",
    description: "Designing and implementing secure cloud solutions using Google Cloud Platform."
  }
];

export function WhatImDoingSection() {
  return (
    <section className={p.section}>
      <div className={p.container}>
        <h2 className={p.heading}>
          What I&apos;m Doing
        </h2>
        <div className={p.grid}>
          {activities.map((activity, index) => (
            <Card
              key={index}
              className={p.card}
            >
              <div className={p.cardInner}>
                <div className={p.iconWrapper}>
                  <Image
                    src={activity.icon}
                    alt={activity.title}
                    width={32}
                    height={32}
                    className={p.icon}
                  />
                </div>
                <div className={p.textWrapper}>
                  <h3 className={p.title}>
                    {activity.title}
                  </h3>
                  <p className={p.description}>
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
