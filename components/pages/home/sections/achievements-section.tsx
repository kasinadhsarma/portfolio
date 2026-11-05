'use client'

import type { AchievementCard } from "@/types/content";
import { certificatesPatterns as p } from "@/lib/responsive/pattrens/home";

interface AchievementsSectionProps {
  achievements: AchievementCard[];
}

export function AchievementsSection({ achievements }: AchievementsSectionProps) {
  return (
    <section className={p.section}>
      <div className={p.container}>
        <div>
          <h2 className={p.heading}>Achievements</h2>
          <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {achievements.map((a) => (
              <li key={a.id} className="flex gap-4 rounded-xl border border-border bg-card/50 p-4">
                {a.icon && (
                  // Plain <img>: CMS storage hosts are not in next.config remotePatterns.
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={a.icon} alt="" loading="lazy" className="h-16 w-16 flex-none rounded-lg object-cover" />
                )}
                <div className="min-w-0">
                  <h3 className="font-semibold leading-snug">{a.title}</h3>
                  {a.date && <p className="text-sm text-muted-foreground">{a.date}</p>}
                  {a.description && <p className="mt-1 text-sm text-muted-foreground">{a.description}</p>}
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
