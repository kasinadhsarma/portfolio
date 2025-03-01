'use client'

import { useRef } from 'react'
import Image from 'next/image'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { cn, storeAndEncodeUrl, safeOpenUrl } from '@/lib/utils'
import { Certificate } from '@/data/certificates'
import { Button } from './button'

interface CertificatesCarouselProps {
  certificates: Certificate[]
  className?: string
}

export function CertificatesCarousel({ certificates, className }: CertificatesCarouselProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  const scroll = (direction: 'left' | 'right') => {
    const container = scrollContainerRef.current
    if (!container) return

    const scrollAmount = container.clientWidth * (direction === 'left' ? -0.8 : 0.8)
    container.scrollBy({ left: scrollAmount, behavior: 'smooth' })
  }

  return (
    <div className={cn("relative w-full group", className)}>
      <div
        ref={scrollContainerRef}
        className="flex gap-6 overflow-x-auto no-scrollbar scroll-smooth"
      >
        {certificates.map((cert, index) => (
          <div
            key={index}
            onClick={() => safeOpenUrl(storeAndEncodeUrl(cert.url))}
            className="flex-none w-48 h-48 relative transition-transform hover:scale-105 cursor-pointer"
          >
            <Image
              src={cert.image}
              alt={cert.title}
              fill
              className="object-contain"
              sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
            />
          </div>
        ))}
      </div>
      
      <Button
        variant="outline"
        size="icon"
        className="absolute left-0 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity"
        onClick={() => scroll('left')}
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>
      
      <Button
        variant="outline"
        size="icon"
        className="absolute right-0 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity"
        onClick={() => scroll('right')}
      >
        <ChevronRight className="h-4 w-4" />
      </Button>
    </div>
  )
}
