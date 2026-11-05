import 'server-only'
import { cmsList } from '@/lib/cms'
import { assetUrl } from '@/lib/cms-assets'
import type {
  CertificateCard,
  AchievementCard,
  ProjectSummary,
  Resume,
  ResumeFileCard,
  HeroContent,
  AboutContent,
  ActivityCard,
  SkillGroup,
} from '@/types/content'

type AssetField = { url?: string | null } | null | undefined

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
    image: assetUrl(r.image as AssetField),
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
      image: assetUrl(r.image as AssetField),
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
    const url = assetUrl(r.file as AssetField)
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

// ─── Static fallbacks ─────────────────────────────────────────────────────────

const DEFAULT_HERO: HeroContent = {
  name: 'Kasinadh Sarma',
  roles: [
    'Cybersecurity Engineer',
    'Penetration Testing Engineer',
    'DevSecOps Engineer',
    'Full Stack Engineer',
  ],
  tagline:
    "Bridging the gap between application security and full-stack engineering to build software that's both usable and hard to break.",
  email: 'kasinadhsarma@gmail.com',
  phone: '+916305953487',
  github: 'https://github.com/kasinadhsarma',
  linkedin: 'https://www.linkedin.com/in/kasinadhsarma',
  twitter: 'https://x.com/kasinadhsarma',
  avatarImage: '/img/me_background.png',
}

const DEFAULT_ABOUT: AboutContent = {
  paragraphs: [
    "I'm a Software Engineer with a B.Tech in Cyber Security from Parul University, specializing in Application Security (AppSec), DevSecOps, and full-stack web development.",
    "I'm Kasinadh Sarma. I build secure, production-grade web applications with Next.js, TypeScript, and Node.js, integrating security into the SDLC by hardening API routes, enforcing strict secrets hygiene in CI/CD pipelines, and maintaining zero-vulnerability dependencies. I also work as a penetration tester, identifying and helping remediate vulnerabilities across web applications and infrastructure.",
  ],
}

const DEFAULT_ACTIVITIES: ActivityCard[] = [
  {
    icon: '/img/icon-design.svg',
    title: 'Web Development',
    description: 'Full-stack development with Next.js, React, Node.js, and TypeScript.',
  },
  {
    icon: '/img/icon-dev.svg',
    title: 'Cybersecurity Research',
    description: 'Advanced security testing, vulnerability assessment, and malware analysis.',
  },
  {
    icon: '/img/icons8-malware-94.png',
    title: 'Penetration Testing',
    description:
      'Identifying and exploiting vulnerabilities in web apps and infrastructure to strengthen security posture.',
  },
  {
    icon: '/img/google-cloud-icon-2048x1646-7admxejz.png',
    title: 'Cloud Architecture',
    description: 'Designing and implementing secure cloud solutions using Google Cloud Platform.',
  },
]

const DEFAULT_SKILLS: SkillGroup[] = [
  {
    category: 'Security & AppSec',
    items: ['Penetration Testing', 'API Hardening', 'Input Validation', 'Secrets Hygiene', 'Dependency Auditing', 'RBAC'],
  },
  {
    category: 'Development',
    items: ['Next.js', 'React', 'Node.js', 'TypeScript', 'Python', 'REST APIs'],
  },
  {
    category: 'Cloud & DevSecOps',
    items: ['Google Cloud Platform', 'Firebase Security Rules', 'Vercel', 'GitHub Actions CI/CD Security'],
  },
  {
    category: 'Languages & Tools',
    items: ['Python', 'TypeScript', 'Dart', 'Bash', 'Git', 'Linux'],
  },
]

// ─── Home page data fetchers ───────────────────────────────────────────────────

/** Hero section content. Falls back to defaults when CMS collection is empty. */
export async function getHomeContent(): Promise<HeroContent> {
  const [record] = await cmsList('hero', { limit: 1 })
  if (!record) return DEFAULT_HERO
  return {
    name: (record.name as string | null) ?? DEFAULT_HERO.name,
    roles: (record.roles as string[] | null) ?? DEFAULT_HERO.roles,
    tagline: (record.tagline as string | null) ?? DEFAULT_HERO.tagline,
    email: (record.email as string | null) ?? DEFAULT_HERO.email,
    phone: (record.phone as string | null) ?? DEFAULT_HERO.phone,
    github: (record.github as string | null) ?? DEFAULT_HERO.github,
    linkedin: (record.linkedin as string | null) ?? DEFAULT_HERO.linkedin,
    twitter: (record.twitter as string | null) ?? DEFAULT_HERO.twitter,
    avatarImage: (record.avatarImage as string | null) ?? DEFAULT_HERO.avatarImage,
  }
}

/** About section content. Falls back to defaults when CMS collection is empty. */
export async function getAboutContent(): Promise<AboutContent> {
  const [record] = await cmsList('about', { limit: 1 })
  if (!record) return DEFAULT_ABOUT
  const paragraphs = record.paragraphs as string[] | null
  return { paragraphs: paragraphs && paragraphs.length > 0 ? paragraphs : DEFAULT_ABOUT.paragraphs }
}

/** Activities ("What I'm Doing") cards. Falls back to defaults when CMS collection is empty. */
export async function getActivities(): Promise<ActivityCard[]> {
  const records = await cmsList('activities', { limit: 20, orderBy: ['order', 'asc'] })
  if (!records.length) return DEFAULT_ACTIVITIES
  return records.map((r) => ({
    icon: (r.icon as string | null) ?? '',
    title: String(r.title ?? ''),
    description: String(r.description ?? ''),
  }))
}

/** Skills section. Falls back to defaults when CMS collection is empty. */
export async function getSkills(): Promise<SkillGroup[]> {
  const records = await cmsList('skills', { limit: 50, orderBy: ['order', 'asc'] })
  if (!records.length) return DEFAULT_SKILLS
  // Group by category, preserving insertion order
  const map = new Map<string, string[]>()
  for (const r of records) {
    const cat = String(r.category ?? 'Other')
    const item = String(r.item ?? '')
    if (!map.has(cat)) map.set(cat, [])
    map.get(cat)!.push(item)
  }
  return Array.from(map.entries()).map(([category, items]) => ({ category, items }))
}

/** Achievements from the CMS. Empty when the collection has no published records. */
export async function getAchievements(): Promise<AchievementCard[]> {
  const records = await cmsList('achievements', { limit: 50, orderBy: ['order', 'asc'] })
  return records.map((r) => ({
    id: String(r.id),
    title: String(r.title ?? ''),
    description: (r.description as string | null) ?? null,
    date: (r.date as string | null) ?? null,
    icon: assetUrl(r.icon as AssetField),
  }))
}
