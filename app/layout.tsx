import React from 'react'
import './globals.css'

export const metadata = {
  title: 'Enhanced Portfolio',
  description: 'A modern developer portfolio built with Next.js',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-900">{children}</body>
    </html>
  )
}
