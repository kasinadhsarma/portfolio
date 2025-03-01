import { Card } from "@/components/ui/card";
import Image from "next/image";
import { cn } from "@/lib/utils";

interface AchievementBadgeProps {
  title: string;
  icon: string;
  description: string;
  date?: string;
  variant?: "small" | "large";
  className?: string;
}

export function AchievementBadge({
  title,
  icon,
  description,
  date,
  variant = "large",
  className,
}: AchievementBadgeProps) {
  return (
    <Card 
      className={cn(
        "group relative overflow-hidden",
        "hover:shadow-2xl hover:shadow-primary/20 transition-all duration-300",
        "dark:bg-zinc-900/50 dark:hover:bg-zinc-900/80",
        "border-border/50 dark:border-border/20",
        "backdrop-blur-sm",
        variant === "small" ? "p-4" : "p-6",
        className
      )}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
      
      <div className="flex gap-4 items-start relative z-10">
        <div className={cn(
          "rounded-xl bg-primary/10 flex items-center justify-center",
          "transition-transform duration-300 group-hover:scale-110",
          variant === "small" ? "w-10 h-10" : "w-14 h-14"
        )}>
          <Image
            src={icon}
            alt={title}
            width={variant === "small" ? 24 : 32}
            height={variant === "small" ? 24 : 32}
            className="object-contain"
          />
        </div>
        
        <div className="flex-1 space-y-1">
          <h3 className={cn(
            "font-semibold leading-tight text-foreground",
            variant === "small" ? "text-sm" : "text-base"
          )}>
            {title}
          </h3>
          
          <p className={cn(
            "text-muted-foreground line-clamp-2",
            variant === "small" ? "text-xs" : "text-sm"
          )}>
            {description}
          </p>
          
          {date && (
            <p className={cn(
              "text-primary font-medium",
              variant === "small" ? "text-xs" : "text-sm"
            )}>
              {date}
            </p>
          )}
        </div>
      </div>
    </Card>
  );
}
