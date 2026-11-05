import { Card } from "@/components/ui/card";
import Image from "next/image";
import { whatImDoingPatterns as p } from "@/lib/responsive/pattrens/home";
import type { ActivityCard } from "@/types/content";

interface WhatImDoingSectionProps {
  activities: ActivityCard[];
}

export function WhatImDoingSection({ activities }: WhatImDoingSectionProps) {
  return (
    <section className={p.section}>
      <div className={p.container}>
        <h2 className={p.heading}>
          What I&apos;m Doing
        </h2>
        <div className={p.grid}>
          {activities.map((activity, index) => (
            <Card key={index} className={p.card}>
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
