"use client"

import React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Menu, Moon, Sun, ChevronUp } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
import { cn } from "@/lib/utils"
import { ResumeDropdown } from "@/components/ui/resume-dropdown"
import { mainNavPatterns as p } from "@/lib/responsive/pattrens/layout"

const routes = [
  { href: "/", label: "About" },
  { href: "/resume", label: "Resume" },
  { href: "/projects", label: "Projects"},
  { href: "https://blogs.kasinadhsarma.in/", label: "Blog", external: true },
  { href: "/research", label: "Research" }
]

const MainNav = () => {
  const pathname = usePathname()
  const [isOpen, setIsOpen] = React.useState(false)
  const [lastScrollY, setLastScrollY] = React.useState(0)
  const [theme, setTheme] = React.useState<"light" | "dark">("dark")

  // Track scroll position (nav itself stays pinned; only drives the scroll-to-top button)
  React.useEffect(() => {
    const handleScroll = () => setLastScrollY(window.scrollY)

    window.addEventListener("scroll", handleScroll, { passive: true })
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  React.useEffect(() => {
    const savedTheme = localStorage.getItem("theme") as "light" | "dark" | null
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches
    const initialTheme = savedTheme || (prefersDark ? "dark" : "light")
    setTheme(initialTheme)
    document.documentElement.classList.toggle("dark", initialTheme === "dark")
  }, [])

  const toggleTheme = () => {
    const newTheme = theme === "dark" ? "light" : "dark"
    setTheme(newTheme)
    localStorage.setItem("theme", newTheme)
    document.documentElement.classList.toggle("dark", newTheme === "dark")
  }

  return (
    <div className={p.root}>
      <Sheet open={isOpen} onOpenChange={setIsOpen}>
        <div className={p.innerRow}>
          {/* Mobile compact pill: menu trigger + theme toggle grouped together */}
          <div className={p.mobilePill}>
            <SheetTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className={p.mobileMenuButton}
              >
                <Menu className="h-5 w-5" />
                <span className="sr-only">Toggle navigation</span>
              </Button>
            </SheetTrigger>
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              className={p.mobileThemeButton}
            >
              {theme === "dark" ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </Button>
          </div>

          <nav className={p.desktopNav}>
            {routes.map((route) => (
            <Link
              key={route.href}
              href={route.href}
              target={route.external ? "_blank" : undefined}
              rel={route.external ? "noopener noreferrer" : undefined}
              className={cn(
                p.desktopLink,
                pathname === route.href && !route.external && p.desktopLinkActive
              )}
            >
              {route.label}
              {route.external && (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className={p.externalIcon}
                >
                  <path d="M7 7h10v10" />
                  <path d="M7 17 17 7" />
                </svg>
              )}
            </Link>
          ))}
          <ResumeDropdown variant="outline" size="sm" />

          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className={p.desktopThemeButton}
          >
            {theme === "dark" ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </Button>
        </nav>
        </div>

        <SheetContent
          side="bottom"
          className={cn(
            p.sheetContent,
            isOpen ? p.sheetContentSlideUp : p.sheetContentSlideDown
          )}
        >
          <nav className={p.mobileNav}>
            {routes.map((route) => (
              <Link
                key={route.href}
                href={route.href}
                target={route.external ? "_blank" : undefined}
                rel={route.external ? "noopener noreferrer" : undefined}
                onClick={() => setIsOpen(false)}
                className={cn(
                  p.mobileLink,
                  pathname === route.href && !route.external && p.mobileLinkActive
                )}
              >
                {route.label}
                {route.external && (
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className={p.mobileExternalIcon}
                  >
                    <path d="M7 7h10v10" />
                    <path d="M7 17 17 7" />
                  </svg>
                )}
              </Link>
            ))}
            <div className={p.mobileResumeWrapper}>
              <ResumeDropdown className="w-full" />
            </div>
          </nav>
        </SheetContent>
      </Sheet>

      {/* Scroll to top button */}
      <Button
        variant="ghost"
        size="icon"
        onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
        className={cn(
          p.scrollTopButton,
          lastScrollY < 100 && p.scrollTopButtonHidden
        )}
      >
        <ChevronUp className="h-5 w-5" />
      </Button>
    </div>
  )
}

export default MainNav
