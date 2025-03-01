"use client";

import { useState, useRef } from "react";
import Image from "next/image";
import { Card } from "@/components/ui/card";
import Link from "next/link";

interface CarouselItem {
  img: string;
  url?: string;
  title?: string;
}

interface CircularCarouselProps {
  items: CarouselItem[];
  radius?: number;
  itemSize?: number;
}

type WrapperProps = {
  href?: string;
  target?: string;
  className?: string;
  style?: React.CSSProperties;
  children: React.ReactNode;
};

const ItemWrapper = ({ href, ...props }: WrapperProps) => {
  if (href) {
    return <Link href={href} {...props} />;
  }
  return <div {...props} />;
};

const CircularCarousel = ({ items, radius = 250, itemSize = 100 }: CircularCarouselProps) => {
  const [rotation, setRotation] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const rotate = (direction: 'left' | 'right') => {
    const step = 360 / items.length;
    setRotation(prev => prev + (direction === 'left' ? step : -step));
  };

  return (
    <div className="relative h-[600px] w-full flex items-center justify-center group">
      <button
        onClick={() => rotate('left')}
        className="absolute left-4 z-10 p-3 rounded-full bg-amber-600/20 hover:bg-amber-600/40 transition-colors"
      >
        ←
      </button>
      <button
        onClick={() => rotate('right')}
        className="absolute right-4 z-10 p-3 rounded-full bg-amber-600/20 hover:bg-amber-600/40 transition-colors"
      >
        →
      </button>

      <div 
        ref={containerRef}
        className="relative w-full h-full transition-transform duration-1000 ease-in-out"
        style={{ transform: `rotate(${rotation}deg)` }}
      >
        {items.map((item, index) => {
          const angle = (360 / items.length) * index;
          const radian = (angle * Math.PI) / 180;
          const x = Math.cos(radian) * radius;
          const y = Math.sin(radian) * radius;

          return (
            <ItemWrapper
              key={index}
              href={item.url}
              target={item.url ? "_blank" : undefined}
              className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 transition-all duration-300"
              style={{
                transform: `translate(${x}px, ${y}px) rotate(${-rotation}deg)`,
                width: `${itemSize}px`,
                height: `${itemSize}px`,
              }}
            >
              <Card className="w-full h-full group/item hover:scale-110 transition-all duration-300 border border-amber-600/20 hover:border-amber-600/50 hover:shadow-2xl hover:shadow-amber-600/10">
                <div className="relative w-full h-full p-2 bg-zinc-900/50 backdrop-blur-sm">
                  <Image
                    src={item.img.startsWith('/') ? item.img : `/img/${item.img}`}
                    alt={item.title || "Badge"}
                    fill
                    className="object-contain p-2 transition-transform duration-300 group-hover/item:scale-110"
                  />
                </div>
                {item.title && (
                  <div className="absolute inset-0 flex items-end justify-center p-2 bg-gradient-to-t from-black/70 to-transparent opacity-0 group-hover/item:opacity-100 transition-opacity">
                    <span className="text-white text-sm text-center font-medium truncate">
                      {item.title}
                    </span>
                  </div>
                )}
              </Card>
            </ItemWrapper>
          );
        })}
      </div>
    </div>
  );
};

export { CircularCarousel };
