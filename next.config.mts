// Files uploaded in the /sethu editor are served by the SethuCMS gateway, so its address must be
// allowed as an image source. Without SETHUCMS_BASE_URL (for example in CI) nothing is added.
const cmsOrigin = (() => {
  try {
    return process.env.SETHUCMS_BASE_URL ? new URL(process.env.SETHUCMS_BASE_URL).origin : null
  } catch {
    return null
  }
})()
const cmsImagePatterns = cmsOrigin
  ? [{ protocol: new URL(cmsOrigin).protocol.replace(':', '') as 'http' | 'https', hostname: new URL(cmsOrigin).hostname }]
  : []

/** @type {import('next').NextConfig} */
const nextConfig = {
  // NOTE: 'output: export' is intentionally removed (server-rendered CMS pages).
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
      ...cmsImagePatterns,
      // TEMPORARY: files not yet copied by scripts/move-assets-to-sethu.mts. Remove these three
      // cdn.sanity.io entries (images, img-src, connect-src) once that script reports 0 left.
      { protocol: 'https', hostname: 'cdn.sanity.io' },
    ],
  },
  experimental: {
    webpackBuildWorker: true,
    parallelServerBuildTraces: true,
    parallelServerCompiles: true,
  },
  async rewrites() {
    // SethuCMS admin is a static Vite build served from public/sethu/.
    return [{ source: '/sethu', destination: '/sethu/index.html' }]
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
      `img-src 'self' data:${cmsOrigin ? ` ${cmsOrigin}` : ''} https://cdn.sanity.io https://komarev.com https://wakatime.com https://www.google.com https://www.google.co.in`,
      "font-src 'self' data: https://cdn.jsdelivr.net",
      "connect-src 'self' https://www.google-analytics.com https://analytics.google.com https://*.google-analytics.com https://stats.g.doubleclick.net https://out-omega-navy.vercel.app",
      "frame-ancestors 'self'",
    ].join('; ')

    const universalHeaders = [
      { key: 'Strict-Transport-Security', value: 'max-age=63072000; includeSubDomains; preload' },
      { key: 'X-Content-Type-Options', value: 'nosniff' },
      { key: 'X-Frame-Options', value: 'SAMEORIGIN' },
      { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
      { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },
    ]

    // The SethuCMS admin at /sethu signs people in with Firebase (Google and
    // GitHub), which loads Google's sign-in script and frame. It needs a wider
    // policy than the rest of the site, so it gets its own and the strict one
    // above keeps applying everywhere else.
    const cmsCsp = [
      "default-src 'self'",
      `script-src 'self' 'unsafe-inline'${isDev ? " 'unsafe-eval'" : ''} https://apis.google.com`,
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob: https://out-omega-navy.vercel.app https://cdn.sanity.io",
      "font-src 'self' data:",
      "connect-src 'self' https://out-omega-navy.vercel.app https://*.googleapis.com https://*.firebaseapp.com https://apis.google.com",
      "frame-src https://*.firebaseapp.com https://accounts.google.com https://apis.google.com https://github.com",
      "frame-ancestors 'self'",
    ].join('; ')

    return [
      {
        source: '/((?!sethu(?:/|$)).*)',
        headers: [...universalHeaders, { key: 'Content-Security-Policy', value: csp }],
      },
      {
        source: '/sethu/:path*',
        headers: [...universalHeaders, { key: 'Content-Security-Policy', value: cmsCsp }],
      },
    ]
  },
}

export default nextConfig
