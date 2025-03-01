import React from "react"
import { Poppins } from "next/font/google"
import type { Metadata } from "next"

import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"
import MainNav from "@/components/main-nav"

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
})

export const metadata: Metadata = {
  title: "Kasinadh Sarma",
  description:
    "Portfolio of Kasinadh Sarma, a cybersecurity enthusiast with a B.Tech in Cyber/Computer Forensics and Counterterrorism from Parui University."
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/img/icon.png" type="image/png" sizes="any" />
        <link href="https://cdn.jsdelivr.net/npm/boxicons@2.0.5/css/boxicons.min.css" rel="stylesheet" />
      </head>
      <body className={`min-h-screen bg-background antialiased ${poppins.className}`}>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
          <div className="relative min-h-screen bg-background">
            <main className="container max-w-6xl py-8 pb-32">
              {children}
            </main>
            <MainNav />
            {/* Views Counter */}
            <div className="fixed bottom-4 right-4 z-50">
              <a
                href="https://github.com/kasinadhsarma/portfolio"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 rounded-lg bg-background/95 p-2 text-sm text-muted-foreground shadow-lg backdrop-blur"
              >
                <img
                  alt="GitHub Views"
                  width={150}
                  height={20}
                  src="https://komarev.com/ghpvc/?username=kasinadhsarma&label=Portfolio+Views"
                />
              </a>
            </div>
          </div>
        </ThemeProvider>
      </body>
    </html>
  )
}
