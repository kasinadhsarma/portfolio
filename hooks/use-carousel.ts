import { useCallback, useEffect, useRef } from 'react';

export function useCarousel(itemCount: number) {
  const carouselRef = useRef<HTMLDivElement>(null);
  const currentRotation = useRef(0);

  const rotate = useCallback((direction: 'left' | 'right') => {
    if (!carouselRef.current) return;
    
    const step = 360 / itemCount;
    currentRotation.current += direction === 'left' ? step : -step;
    
    carouselRef.current.style.transform = `rotate(${currentRotation.current}deg)`;
    
    // Rotate items in opposite direction to keep them upright
    const items = carouselRef.current.children;
    Array.from(items).forEach((item: Element) => {
      (item as HTMLElement).style.transform = `rotate(${-currentRotation.current}deg)`;
    });
  }, [itemCount]);

  return {
    carouselRef,
    rotate
  };
}
