"use client"

import React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Menu, Moon, Sun, ChevronUp } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
import { cn } from "@/lib/utils"
import { ResumeDropdown } from "@/components/ui/resume-dropdown"

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
  const [isVisible, setIsVisible] = React.useState(true)
  const [lastScrollY, setLastScrollY] = React.useState(0)
  const [theme, setTheme] = React.useState<"light" | "dark">("dark")

  // Handle scroll behavior
  React.useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY
      setIsVisible(lastScrollY > currentScrollY || currentScrollY < 100)
      setLastScrollY(currentScrollY)
    }

    window.addEventListener("scroll", handleScroll, { passive: true })
    return () => window.removeEventListener("scroll", handleScroll)
  }, [lastScrollY])

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
    <div className={cn(
      "fixed left-0 right-0 z-50 flex justify-center p-4",
      "bottom-0 transform transition-all duration-500 ease-in-out",
      isVisible ? "translate-y-0 opacity-100" : "translate-y-full opacity-0"
    )}>
      <Sheet open={isOpen} onOpenChange={setIsOpen}>
        <SheetTrigger asChild>
          <Button 
            variant="ghost" 
            size="icon" 
            className={cn(
              "md:hidden text-primary absolute left-4",
              "hover:bg-primary/10 transition-all duration-200"
            )}
          >
            <Menu className="h-6 w-6" />
            <span className="sr-only">Toggle navigation</span>
          </Button>
        </SheetTrigger>
        <SheetContent 
          side="bottom" 
          className={cn(
            "w-full bg-card/95 border-none rounded-t-3xl",
            "transform transition-all duration-500 ease-in-out",
            isOpen ? "animate-slide-up" : "animate-slide-down"
          )}
        >
          <nav className={cn(
            "flex flex-col gap-2",
            "transform transition-all duration-300",
            "data-[state=open]:translate-y-0 data-[state=open]:opacity-100",
            "data-[state=closed]:translate-y-4 data-[state=closed]:opacity-0"
          )}>
            {routes.map((route) => (
              <Link
                key={route.href}
                href={route.href}
                target={route.external ? "_blank" : undefined}
                rel={route.external ? "noopener noreferrer" : undefined}
                onClick={() => setIsOpen(false)}
                className={cn(
                  "text-lg font-medium px-4 py-3 text-primary",
                  "transition-all duration-200 rounded-lg",
                  "hover:text-primary hover:bg-primary/10",
                  "active:scale-95",
                  pathname === route.href && !route.external && "text-primary/90 bg-primary/20"
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
                    className="ml-2 inline-block h-4 w-4"
                  >
                    <path d="M7 7h10v10" />
                    <path d="M7 17 17 7" />
                  </svg>
                )}
              </Link>
            ))}
            <div className="px-4 py-2">
              <ResumeDropdown className="w-full" />
            </div>
          </nav>
        </SheetContent>
      </Sheet>

      <div className="flex items-center gap-4">
        <nav className={cn(
          "hidden md:flex md:gap-8 bg-background/95 px-8 py-3",
          "rounded-full shadow-lg items-center backdrop-blur-sm",
          "border border-border/50",
          "transform transition-all duration-500 ease-in-out",
          isVisible ? "translate-y-0 opacity-100" : "translate-y-8 opacity-0",
          "hover:shadow-xl hover:bg-background/98",
          "hover:border-primary/20",
          "hover:scale-[1.02]"
        )}>
          {routes.map((route) => (
            <Link
              key={route.href}
              href={route.href}
              target={route.external ? "_blank" : undefined}
              rel={route.external ? "noopener noreferrer" : undefined}
              className={cn(
                "text-sm font-medium text-muted-foreground",
                "transition-all duration-200",
                "hover:text-foreground hover:scale-105",
                "active:scale-95",
                pathname === route.href && !route.external && "text-foreground font-bold"
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
                  className="ml-1 inline-block h-3 w-3"
                >
                  <path d="M7 7h10v10" />
                  <path d="M7 17 17 7" />
                </svg>
              )}
            </Link>
          ))}
          <ResumeDropdown variant="outline" size="sm" />
        </nav>

        <Button
          variant="ghost"
          size="icon"
          onClick={toggleTheme}
          className={cn(
            "text-primary",
            "transition-all duration-200",
            "hover:text-primary/90 hover:bg-primary/10",
            "active:scale-95"
          )}
        >
          {theme === "dark" ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
        </Button>
      </div>
      
      {/* Scroll to top button */}
      <Button
        variant="ghost"
        size="icon"
        onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
        className={cn(
          "fixed bottom-20 right-4 text-primary",
          "transition-all duration-200",
          "hover:text-primary/90 hover:bg-primary/10",
          "md:bottom-24",
          lastScrollY < 100 && "opacity-0 pointer-events-none"
        )}
      >
        <ChevronUp className="h-5 w-5" />
      </Button>
    </div>
  )
}

export default MainNav
