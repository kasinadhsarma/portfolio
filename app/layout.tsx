import React from "react"
import { Poppins } from "next/font/google"
import type { Metadata } from "next"
import "./globals.css"

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
})

export const metadata: Metadata = {
  title: "Kasinadh Sarma",
  description:
    "Portfolio of Kasinadh Sarma, a cybersecurity enthusiast with a B.Tech in Cyber/Computer Forensics and Counterterrorism from Parui University.",
}

/**
 * Root layout — intentionally minimal so /studio route doesn't inherit
 * portfolio-specific UI (nav, theme provider, etc.).
 * Portfolio-specific layout lives in app/(main)/layout.tsx.
 */
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
        {children}
      </body>
    </html>
  )
}

