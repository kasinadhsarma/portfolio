"use client"

import type React from "react"
import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"

const ThemeToggle: React.FC = () => {
  const { theme, setTheme } = useTheme()
  const isDark = theme === "dark"

  return (
    <div
      role="button"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className={`relative flex h-8 w-16 cursor-pointer items-center rounded-full p-1 transition-colors duration-300 ${
        isDark ? "bg-slate-700" : "bg-amber-100"
      }`}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          setTheme(isDark ? "light" : "dark")
        }
      }}
    >
      <div
        className={`absolute h-6 w-6 rounded-full transition-transform duration-300 ${
          isDark ? "translate-x-8 bg-slate-900" : "translate-x-0 bg-amber-400"
        }`}
      />
      <Sun className={`absolute left-1.5 h-4 w-4 text-amber-400 transition-opacity ${isDark ? "opacity-0" : "opacity-100"}`} />
      <Moon className={`absolute right-1.5 h-4 w-4 text-slate-200 transition-opacity ${isDark ? "opacity-100" : "opacity-0"}`} />
      <span className="sr-only">Toggle theme</span>
    </div>
  )
}

export default ThemeToggle
