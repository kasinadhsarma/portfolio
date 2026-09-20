"use client"

import type React from "react"
import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"
import { cn } from "@/lib/utils"
import { themeTogglePatterns as p } from "@/lib/responsive/pattrens/ui"

const ThemeToggle: React.FC = () => {
  const { theme, setTheme } = useTheme()
  const isDark = theme === "dark"

  return (
    <div
      role="button"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className={cn(p.track, isDark ? p.trackDark : p.trackLight)}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          setTheme(isDark ? "light" : "dark")
        }
      }}
    >
      <div
        className={cn(p.thumb, isDark ? p.thumbDark : p.thumbLight)}
      />
      <Sun className={cn(p.sunIcon, isDark ? p.iconHidden : p.iconVisible)} />
      <Moon className={cn(p.moonIcon, isDark ? p.iconVisible : p.iconHidden)} />
      <span className="sr-only">Toggle theme</span>
    </div>
  )
}

export default ThemeToggle
