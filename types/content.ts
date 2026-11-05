export interface ProjectSummary {
  id: string
  title: string
  slug: string
  description?: string | null
  image?: string | null
  technologies?: string[] | null
  category: string[]
  github?: string | null
  liveUrl?: string | null
  featured: boolean
  publishedAt: string
}

export interface CertificateCard {
  id: string
  title: string
  issuer: string
  date?: string | null
  image?: string | null
  url?: string | null
  category: 'featured' | 'cloud' | 'work' | 'practical'
}

export interface ResumeFileCard {
  label: string
  url: string
}

export interface EducationEntry {
  institution: string
  period: string
  description?: string
}

export interface ExperienceEntry {
  title: string
  organization: string
  period: string
  current?: boolean
  highlights?: string[]
}

export interface ResumeProjectEntry {
  title: string
  subtitle?: string
  highlights?: string[]
}

export interface TrainingEntry {
  title: string
  organization: string
  period: string
  description?: string
}

export interface Resume {
  education: EducationEntry[]
  experience: ExperienceEntry[]
  projects: ResumeProjectEntry[]
  training: TrainingEntry[]
}

export interface HeroContent {
  name: string
  roles: string[]
  tagline: string
  email: string
  phone: string
  github: string
  linkedin: string
  twitter: string
  avatarImage: string
}

export interface AboutContent {
  paragraphs: string[]
}

export interface ActivityCard {
  icon: string
  title: string
  description: string
}

export interface SkillGroup {
  category: string
  items: string[]
}