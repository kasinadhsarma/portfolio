import { cn } from "@/lib/utils";
import { useScrollAnimation } from "@/hooks/use-scroll-animation";
import { scrollSectionPatterns as p } from "@/lib/responsive/pattrens/ui";

interface ScrollSectionProps {
  children: React.ReactNode;
  direction?: "up" | "down" | "left" | "right";
  delay?: number;
  className?: string;
}

export function ScrollSection({
  children,
  direction = "up",
  delay = 0,
  className
}: ScrollSectionProps) {
  const { ref, isVisible } = useScrollAnimation();

  const directionStyles = {
    up: p.directionUp,
    down: p.directionDown,
    left: p.directionLeft,
    right: p.directionRight,
  };

  return (
    <div
      ref={ref}
      className={cn(
        p.base,
        isVisible ? p.visible : `${directionStyles[direction]} ${p.hidden}`,
        className
      )}
      style={{
        transitionDelay: `${delay}ms`
      }}
    >
      {children}
    </div>
  );
}
