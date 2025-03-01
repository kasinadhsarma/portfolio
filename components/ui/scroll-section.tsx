import { cn } from "@/lib/utils";
import { useScrollAnimation } from "@/hooks/use-scroll-animation";

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
    up: "translate-y-8",
    down: "-translate-y-8",
    left: "translate-x-8",
    right: "-translate-x-8"
  };

  return (
    <div
      ref={ref}
      className={cn(
        "transform transition-all duration-700 ease-out",
        isVisible ? "translate-y-0 translate-x-0 opacity-100" : `${directionStyles[direction]} opacity-0`,
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
