import type React from "react"
import MainNav from "@/components/main-nav"
import ThemeToggle from "@/components/theme-toggle"

const SiteHeader: React.FC = () => {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center">
        <MainNav />
        <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}

export default SiteHeader
