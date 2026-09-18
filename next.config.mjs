/** @type {import('next').NextConfig} */
const nextConfig = {
  // NOTE: 'output: export' is intentionally removed.
  // Sanity Studio (/studio route) requires server-side rendering and is
  // incompatible with full static export mode.
  basePath: '',
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'komarev.com',
        pathname: '/ghpvc/**',
      },
      {
        protocol: 'https',
        hostname: 'cdn.sanity.io',
      },
    ],
  },
  experimental: {
    webpackBuildWorker: true,
    parallelServerBuildTraces: true,
    parallelServerCompiles: true,
  },
  async headers() {
    // 'unsafe-eval' is only needed for Next.js dev-mode tooling (Fast
    // Refresh / stack-trace reconstruction) — React never calls eval() in
    // production, so production stays on the stricter policy.
    const isDev = process.env.NODE_ENV !== 'production'
    const csp = [
      "default-src 'self'",
      `script-src 'self' 'unsafe-inline'${isDev ? " 'unsafe-eval'" : ''} https://www.googletagmanager.com https://va.vercel-scripts.com`,
      "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
      "img-src 'self' data: https://cdn.sanity.io https://komarev.com https://wakatime.com https://www.google.com https://www.google.co.in",
      "font-src 'self' data: https://cdn.jsdelivr.net",
      "connect-src 'self' https://www.google-analytics.com https://analytics.google.com https://*.google-analytics.com https://stats.g.doubleclick.net https://cdn.sanity.io",
      "frame-ancestors 'self'",
    ].join('; ')

    const universalHeaders = [
      { key: 'Strict-Transport-Security', value: 'max-age=63072000; includeSubDomains; preload' },
      { key: 'X-Content-Type-Options', value: 'nosniff' },
      { key: 'X-Frame-Options', value: 'SAMEORIGIN' },
      { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
      { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },
    ]

    return [
      // Applied everywhere, including /studio — none of these constrain what
      // Sanity Studio's admin UI can load, they just harden transport/framing.
      { source: '/:path*', headers: universalHeaders },
      // CSP is scoped to the public site only. Sanity Studio is a heavy
      // client app (inline styles, workers, blob URLs) that a strict CSP
      // built for the public pages would likely break — give it none rather
      // than guess at a permissive-enough policy for internals we don't own.
      { source: '/((?!studio).*)', headers: [{ key: 'Content-Security-Policy', value: csp }] },
    ]
  },
}

export default nextConfig
