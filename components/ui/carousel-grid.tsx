import React, { useRef } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface CarouselGridProps {
  children: React.ReactNode[];
  gap?: 'small' | 'medium' | 'large';
  itemsPerRow?: { mobile?: number; tablet?: number; desktop?: number; };
  className?: string;
}

export function CarouselGrid({
  children,
  gap = 'medium',
  itemsPerRow = { mobile: 2, tablet: 3, desktop: 4 },
  className
}: CarouselGridProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = React.useState(false);
  const [canScrollRight, setCanScrollRight] = React.useState(true);

  const checkScroll = () => {
    if (scrollRef.current) {
      const { scrollLeft, scrollWidth, clientWidth } = scrollRef.current;
      setCanScrollLeft(scrollLeft > 0);
      setCanScrollRight(scrollLeft < scrollWidth - clientWidth - 10);
    }
  };

  React.useEffect(() => {
    checkScroll();
    window.addEventListener('resize', checkScroll);
    return () => window.removeEventListener('resize', checkScroll);
  }, [children]);

  const scroll = (direction: 'left' | 'right') => {
    if (scrollRef.current) {
      const scrollAmount = scrollRef.current.clientWidth * 0.8;
      scrollRef.current.scrollBy({
        left: direction === 'left' ? -scrollAmount : scrollAmount,
        behavior: 'smooth'
      });
    }
  };

  const gapClass = {
    small: 'gap-2 md:gap-3',
    medium: 'gap-4 md:gap-6',
    large: 'gap-6 md:gap-8'
  }[gap];

  const gridClass = cn(
    'grid grid-flow-col auto-cols-max',
    gapClass,
    `grid-rows-${Math.ceil(children.length / (itemsPerRow.desktop || 4))}`
  );

  return (
    <div className={cn("relative group", className)}>
      <div
        ref={scrollRef}
        className="overflow-x-auto hide-scrollbar snap-x snap-mandatory"
        onScroll={checkScroll}
      >
        <div className={gridClass}>
          {children.map((child, index) => (
            <div key={index} className="snap-start">
              {child}
            </div>
          ))}
        </div>
      </div>

      {/* Navigation Buttons */}
      <Button
        variant="ghost"
        size="icon"
        className={cn(
          "absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1/2",
          "opacity-0 group-hover:opacity-100 transition-opacity duration-200",
          "bg-background/80 backdrop-blur-sm hover:bg-background",
          "disabled:opacity-0",
          "hidden md:flex"
        )}
        onClick={() => scroll('left')}
        disabled={!canScrollLeft}
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>

      <Button
        variant="ghost"
        size="icon"
        className={cn(
          "absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2",
          "opacity-0 group-hover:opacity-100 transition-opacity duration-200",
          "bg-background/80 backdrop-blur-sm hover:bg-background",
          "disabled:opacity-0",
          "hidden md:flex"
        )}
        onClick={() => scroll('right')}
        disabled={!canScrollRight}
      >
        <ChevronRight className="h-4 w-4" />
      </Button>
    </div>
  );
}
