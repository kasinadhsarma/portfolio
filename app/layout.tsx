import React from "react"
import { Poppins } from "next/font/google"
import type { Metadata } from "next"
import "./globals.css"
import MainNav from "@/components/main-nav"
import { ThemeProvider } from "@/components/theme-provider"
import { Toaster } from "@/components/ui/toaster"
import { headers } from "next/headers"

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
})

export const metadata: Metadata = {
  title: "Kasinadh Sarma",
  description:
    "Portfolio of Kasinadh Sarma, a cybersecurity enthusiast with a B.Tech in Cyber/Computer Forensics and Counterterrorism from Parui University.",
}

async function LayoutWrapper({ children }: { children: React.ReactNode }) {
  const headersList = await headers()
  const pathname = headersList.get("x-pathname") ?? ""
  const isStudio = pathname.startsWith("/studio")

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      {!isStudio && <MainNav />}
      {children}
      {!isStudio && <Toaster />}
    </ThemeProvider>
  )
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
        {/* Google Analytics */}
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-XVCV1HCLCW"></script>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              window.dataLayer = window.dataLayer || [];
              function gtag(){dataLayer.push(arguments);}
              gtag('js', new Date());
              gtag('config', 'G-XVCV1HCLCW');
            `,
          }}
        />
      </head>
      <body className={`min-h-screen bg-background antialiased ${poppins.className}`}>
        <LayoutWrapper>{children}</LayoutWrapper>
      </body>
    </html>
  )
}