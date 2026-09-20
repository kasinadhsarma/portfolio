// Runs after `next build`. Obfuscates only the client JS chunks that belong
// exclusively to the public pages (/, /projects, /research, /resume) — never
// the Turbopack runtime loader, and never anything also used by /studio
// (Sanity Studio's own multi-megabyte bundle, and shared React/framework
// chunks). Turbopack content-hashes chunk filenames on every build, so the
// target set is computed fresh each run from the client-reference manifests
// rather than hardcoded.
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs'
import path from 'node:path'
import JavaScriptObfuscator from 'javascript-obfuscator'
import type { ObfuscatorOptions } from 'javascript-obfuscator'

const { obfuscate } = JavaScriptObfuscator

const ROOT = process.cwd()
const NEXT_DIR = path.join(ROOT, '.next')
const APP_SERVER_DIR = path.join(NEXT_DIR, 'server', 'app')

const PUBLIC_ROUTES = ['/page', '/resume/page', '/projects/page', '/research/page']
const STUDIO_ROUTE_SUFFIX = '/studio/[[...tool]]/page'

interface RscManifestModule {
  chunks?: string[]
}

interface RscManifest {
  clientModules?: Record<string, RscManifestModule>
}

interface ExtractedManifest {
  route: string | null
  chunks: string[]
}

function findManifestFiles(dir: string): string[] {
  const results: string[] = []
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry)
    const stat = statSync(full)
    if (stat.isDirectory()) {
      results.push(...findManifestFiles(full))
    } else if (entry.endsWith('_client-reference-manifest.js')) {
      results.push(full)
    }
  }
  return results
}

function extractChunks(manifestPath: string): ExtractedManifest {
  const source = readFileSync(manifestPath, 'utf8')
  const match = source.match(/globalThis\.__RSC_MANIFEST\["([^"]+)"\]\s*=\s*(\{[\s\S]*\});?\s*$/)
  if (!match) return { route: null, chunks: [] }

  const [, route, jsonText] = match
  const manifest = JSON.parse(jsonText) as RscManifest
  const chunks = new Set<string>()

  for (const mod of Object.values(manifest.clientModules ?? {})) {
    for (const chunk of mod.chunks ?? []) {
      const cleaned = chunk.replace(/^\/?_next\//, '')
      if (cleaned.startsWith('static/chunks/')) chunks.add(cleaned)
    }
  }

  return { route, chunks: [...chunks] }
}

function main(): void {
  const manifestFiles = findManifestFiles(APP_SERVER_DIR)

  const publicChunks = new Set<string>()
  const studioChunks = new Set<string>()

  for (const file of manifestFiles) {
    const { route, chunks } = extractChunks(file)
    if (!route) continue

    if (route === STUDIO_ROUTE_SUFFIX) {
      chunks.forEach((c) => studioChunks.add(c))
    } else if (PUBLIC_ROUTES.includes(route)) {
      chunks.forEach((c) => publicChunks.add(c))
    }
  }

  const targets = [...publicChunks].filter(
    (chunk) => !studioChunks.has(chunk) && !path.basename(chunk).startsWith('turbopack-')
  )

  if (targets.length === 0) {
    console.log('[obfuscate-build] No public-only chunks found — skipping.')
    return
  }

  const obfuscatorOptions: ObfuscatorOptions = {
    compact: true,
    simplify: true,
    controlFlowFlattening: false,
    deadCodeInjection: false,
    debugProtection: false,
    disableConsoleOutput: false,
    identifierNamesGenerator: 'hexadecimal',
    renameGlobals: false,
    selfDefending: false,
    splitStrings: false,
    stringArray: true,
    stringArrayEncoding: ['base64'],
    stringArrayThreshold: 0.75,
    transformObjectKeys: false,
    numbersToExpressions: false,
    unicodeEscapeSequence: false,
  }

  let totalBefore = 0
  let totalAfter = 0

  for (const relChunk of targets) {
    const fullPath = path.join(NEXT_DIR, relChunk)
    let source: string
    try {
      source = readFileSync(fullPath, 'utf8')
    } catch {
      console.warn(`[obfuscate-build] Skipping missing file: ${relChunk}`)
      continue
    }

    const result = obfuscate(source, obfuscatorOptions).getObfuscatedCode()
    writeFileSync(fullPath, result, 'utf8')

    totalBefore += Buffer.byteLength(source)
    totalAfter += Buffer.byteLength(result)
    console.log(`[obfuscate-build] Obfuscated ${relChunk}`)
  }

  console.log(
    `[obfuscate-build] Done. ${targets.length} chunk(s), ${(totalBefore / 1024).toFixed(1)}KB -> ${(totalAfter / 1024).toFixed(1)}KB`
  )
}

main()
