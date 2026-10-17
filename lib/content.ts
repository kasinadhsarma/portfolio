import 'server-only'
import { cmsList } from '@/lib/cms'
import { assetUrl } from '@/lib/cms-assets'
import type { CertificateCard, ProjectSummary, Resume, ResumeFileCard } from '@/types/content'

type AssetField = { asset?: string | null } | null | undefined

const CERTIFICATE_CATEGORY_ORDER = ['featured', 'cloud', 'work', 'practical']

export async function getProjects(): Promise<ProjectSummary[]> {
  // Sorted here because the gateway can't order this collection server-side.
  const records = await cmsList('projects', { limit: 100 })
  records.sort((a, b) => String(b.publishedAt ?? '').localeCompare(String(a.publishedAt ?? '')))
  return records.map((r) => ({
    id: String(r.id),
    title: String(r.title ?? ''),
    slug: String(r.slug ?? ''),
    description: (r.description as string | null) ?? null,
    image: assetUrl((r.image as AssetField)?.asset, { width: 600, height: 400 }),
    technologies: (r.technologies as string[] | null) ?? null,
    category: (r.category as string[] | null) ?? [],
    github: (r.github as string | null) ?? null,
    liveUrl: (r.liveUrl as string | null) ?? null,
    featured: Boolean(r.featured),
    publishedAt: String(r.publishedAt ?? ''),
  }))
}

export async function getCertificates(): Promise<CertificateCard[]> {
  const records = await cmsList('certificates', { limit: 100 })
  return records
    .map((r) => ({
      id: String(r.id),
      title: String(r.title ?? ''),
      issuer: String(r.issuer ?? ''),
      date: (r.date as string | null) ?? null,
      image: assetUrl((r.image as AssetField)?.asset, { width: 400 }),
      url: (r.url as string | null) ?? null,
      category: r.category as CertificateCard['category'],
      order: typeof r.order === 'number' ? r.order : 0,
    }))
    .sort(
      (a, b) =>
        CERTIFICATE_CATEGORY_ORDER.indexOf(a.category) - CERTIFICATE_CATEGORY_ORDER.indexOf(b.category) ||
        a.order - b.order,
    )
    .map(({ order: _order, ...card }) => card)
}

export async function getResumeFiles(): Promise<ResumeFileCard[]> {
  const records = await cmsList('resume_files', { limit: 20 })
  records.sort((a, b) => Number(a.order ?? 0) - Number(b.order ?? 0))
  return records.flatMap((r) => {
    const url = assetUrl((r.file as AssetField)?.asset)
    return url ? [{ label: String(r.label ?? ''), url }] : []
  })
}

export async function getResume(): Promise<Resume | null> {
  const [record] = await cmsList('resume', { limit: 1 })
  if (!record) return null
  return {
    education: (record.education as Resume['education'] | null) ?? [],
    experience: (record.experience as Resume['experience'] | null) ?? [],
    projects: (record.projects as Resume['projects'] | null) ?? [],
    training: (record.training as Resume['training'] | null) ?? [],
  }
}
