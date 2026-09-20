'use client'

import { useRef } from 'react'
import Image from 'next/image'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { cn, storeAndEncodeUrl, safeOpenUrl } from '@/lib/utils'
import { SanityCertificateCard } from '@/types/sanity'
import { certificatesCarouselPatterns as p } from '@/lib/responsive/pattrens/ui'
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
    <div className={cn(p.root, className)}>
      <div
        ref={scrollContainerRef}
        className={p.scrollRow}
      >
        {certificates.map((cert, index) => (
          <div
            key={cert._id}
            onClick={() => cert.url && safeOpenUrl(storeAndEncodeUrl(cert.url))}
            className={cn(
              p.item,
              cert.url ? p.itemClickable : p.itemStatic
            )}
          >
            {cert.image && (
              <Image
                src={cert.image}
                alt={cert.title}
                fill
                className={p.image}
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
        className={p.navLeft}
        onClick={() => scroll('left')}
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>

      <Button
        variant="outline"
        size="icon"
        className={p.navRight}
        onClick={() => scroll('right')}
      >
        <ChevronRight className="h-4 w-4" />
      </Button>
    </div>
  )
}
