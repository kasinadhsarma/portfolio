"use client"

import React from "react"
import { Info } from "lucide-react"
import { cn } from "@/lib/utils"
import { topBarPatterns as p } from "@/lib/responsive/pattrens/layout"

const MESSAGE =
  "Open to full-stack, cybersecurity, and penetration testing opportunities — reach out via the contact section."

// Root font-size steps (%). Tailwind sizes are rem-based, so this scales the whole site.
export const FONT_SCALES = [87.5, 100, 112.5, 125] as const
export const DEFAULT_FONT_SCALE_INDEX = 1
export const FONT_SCALE_STORAGE_KEY = "font-scale"

const pad = (n: number) => String(n).padStart(2, "0")

const formatNow = (d: Date) =>
  `${pad(d.getDate())}-${pad(d.getMonth() + 1)}-${d.getFullYear()} | ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`

const TopBar = () => {
  // null until mounted so server and client render the same markup
  const [now, setNow] = React.useState<string | null>(null)
  const [scaleIndex, setScaleIndex] = React.useState(DEFAULT_FONT_SCALE_INDEX)

  React.useEffect(() => {
    setNow(formatNow(new Date()))
    const id = window.setInterval(() => setNow(formatNow(new Date())), 1000)
    return () => window.clearInterval(id)
  }, [])

  React.useEffect(() => {
    try {
      const saved = Number(localStorage.getItem(FONT_SCALE_STORAGE_KEY))
      if (Number.isInteger(saved) && saved >= 0 && saved < FONT_SCALES.length) setScaleIndex(saved)
    } catch {}
  }, [])

  const applyScale = (index: number) => {
    setScaleIndex(index)
    document.documentElement.style.fontSize = `${FONT_SCALES[index]}%`
    try {
      localStorage.setItem(FONT_SCALE_STORAGE_KEY, String(index))
    } catch {}
  }

  return (
    <div className={p.root} role="region" aria-label="Site announcement">
      <div className={p.inner}>
        <p className={p.message}>
          <Info className={p.messageIcon} aria-hidden="true" />
          <span className={p.messageText} title={MESSAGE}>{MESSAGE}</span>
        </p>
        <div className={p.controls}>
          <time className={p.clock} suppressHydrationWarning>{now ?? " "}</time>
          <div className={p.fontGroup} role="group" aria-label="Text size">
            <button
              type="button"
              className={p.fontButton}
              onClick={() => applyScale(scaleIndex - 1)}
              disabled={scaleIndex === 0}
              aria-label="Decrease text size"
            >
              A-
            </button>
            <button
              type="button"
              className={cn(p.fontButton, scaleIndex === DEFAULT_FONT_SCALE_INDEX && p.fontButtonActive)}
              onClick={() => applyScale(DEFAULT_FONT_SCALE_INDEX)}
              aria-label="Reset text size"
            >
              A
            </button>
            <button
              type="button"
              className={p.fontButton}
              onClick={() => applyScale(scaleIndex + 1)}
              disabled={scaleIndex === FONT_SCALES.length - 1}
              aria-label="Increase text size"
            >
              A+
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TopBar
