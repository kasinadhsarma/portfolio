'use client'

import { Card, CardHeader, CardTitle } from "@/components/ui/card"
import { storeAndEncodeUrl, safeOpenUrl } from "@/lib/utils"
import { useEffect, useState } from "react"

interface PublicationCardProps {
  title: string
  url: string
  isLocal?: boolean
}

export function PublicationCard({ title, url, isLocal = false }: PublicationCardProps) {
  const [encodedUrl, setEncodedUrl] = useState<string>('')

  useEffect(() => {
    setEncodedUrl(storeAndEncodeUrl(url))
  }, [url])

  const handleClick = () => {
    if (isLocal && !url.startsWith('/pdf/')) {
      console.error('PDF file not found:', url)
      alert('PDF file not available')
      return
    }
    safeOpenUrl(encodedUrl)
  }

  return (
    <Card className="hover:bg-accent/50 transition-colors cursor-pointer" onClick={handleClick}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg hover:text-primary transition-colors">
            {title}
          </CardTitle>
        </div>
      </CardHeader>
    </Card>
  )
}
