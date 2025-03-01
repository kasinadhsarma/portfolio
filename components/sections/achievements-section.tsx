import { AchievementBadge } from "@/components/ui/achievement-badge";
import { CarouselGrid } from "@/components/ui/carousel-grid";
import { achievements, certifications, badges } from "@/lib/data";
import { useScrollAnimation } from "@/hooks/use-scroll-animation";
import { cn } from "@/lib/utils";
import Link from "next/link";

export function AchievementsSection() {
  const { ref, isVisible } = useScrollAnimation();

  return (
    <section
      ref={ref}
      className={cn(
        "py-16 transform transition-all duration-700",
        isVisible ? "translate-y-0 opacity-100" : "translate-y-8 opacity-0"
      )}
    >
      {/* Featured Certifications */}
      <div className="mb-16">
        <h2 className="text-2xl font-semibold mb-8 bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
          Featured Certifications
        </h2>
        <div className="grid gap-6 md:grid-cols-2">
          {certifications.map((cert, index) => (
            <Link key={index} href={cert.url} target="_blank" rel="noopener noreferrer">
              <AchievementBadge
                title={cert.title}
                icon={cert.icon}
                description={`${cert.issuer} • ${cert.date}`}
                className="hover:scale-[1.02] transition-transform duration-300"
              />
            </Link>
          ))}
        </div>
      </div>

      {/* Google Cloud Badges */}
      <div className="mb-16">
        <h2 className="text-2xl font-semibold mb-8 bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
          Google Cloud Badges
        </h2>
        <CarouselGrid gap="medium" itemsPerRow={{ mobile: 2, tablet: 3, desktop: 5 }}>
          {badges.map((badge, index) => (
            <Link key={index} href={badge.url} target="_blank" rel="noopener noreferrer">
              <AchievementBadge
                variant="small"
                title={badge.title}
                icon={badge.icon}
                description="Google Cloud Platform"
                className="w-[160px] hover:scale-105 transition-transform duration-300"
              />
            </Link>
          ))}
        </CarouselGrid>
      </div>

      {/* Additional Achievements */}
      <div>
        <h2 className="text-2xl font-semibold mb-8 bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
          Additional Achievements
        </h2>
        <div className="grid gap-6 md:grid-cols-2">
          {achievements.map((achievement, index) => (
            <AchievementBadge
              key={index}
              title={achievement.title}
              icon={achievement.icon}
              description={achievement.description}
              date={achievement.date}
              className="hover:scale-[1.02] transition-transform duration-300"
            />
          ))}
        </div>
      </div>
    </section>
  );
}
