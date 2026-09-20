'use client'

import { useRef } from 'react'
import Image from 'next/image'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { cn, storeAndEncodeUrl, safeOpenUrl } from '@/lib/utils'
import { SanityCertificateCard } from '@/types/sanity'
import { Button } from './button'

interface CertificatesCarouselProps {
  certificates: SanityCertificateCard[]
  className?: string
  priorityFirst?: boolean
}

export function CertificatesCarousel({ certificates, className, priorityFirst }: CertificatesCarouselProps) {
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
            key={cert._id}
            onClick={() => cert.url && safeOpenUrl(storeAndEncodeUrl(cert.url))}
            className={cn(
              "flex-none w-48 h-48 relative rounded-xl border border-border/40 bg-card p-4 shadow-sm transition-transform hover:scale-105 hover:shadow-md",
              cert.url ? "cursor-pointer" : "cursor-default"
            )}
          >
            {cert.image && (
              <Image
                src={cert.image}
                alt={cert.title}
                fill
                className="object-contain p-2"
                sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                priority={priorityFirst && index === 0}
              />
            )}
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
