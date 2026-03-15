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
}

export default nextConfig
