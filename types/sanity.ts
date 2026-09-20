import { PortableTextBlock } from 'sanity'

export interface SanityProject {
  _id: string
  _type: 'project'
  _createdAt: string
  _updatedAt: string
  title: string
  slug: {
    current: string
  }
  description?: string
  longDescription?: PortableTextBlock[]
  image?: {
    asset: {
      _ref: string
      _type: 'reference'
    }
    alt?: string
  }
  gallery?: Array<{
    asset: {
      _ref: string
      _type: 'reference'
    }
    alt?: string
  }>
  technologies?: string[]
  category: Array<'ai' | 'web' | 'cybersecurity' | 'database' | 'cloud' | 'mobile' | 'desktop' | 'other'>
  status: 'development' | 'completed' | 'on-hold' | 'archived'
  github?: string
  liveUrl?: string
  featured: boolean
  startDate?: string
  endDate?: string
  team?: Array<{
    name: string
    role: string
    url?: string
  }>
  publishedAt: string
}

export interface SanityProjectCard {
  _id: string
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

export interface SanityCertificate {
  _id: string
  _type: 'certificate'
  title: string
  issuer: string
  date?: string
  image?: {
    asset: {
      _ref: string
      _type: 'reference'
    }
    alt?: string
  }
  url?: string
  category: 'featured' | 'cloud' | 'work' | 'practical'
  order?: number
}

export interface SanityCertificateCard {
  _id: string
  title: string
  issuer: string
  date?: string | null
  image?: string | null
  url?: string | null
  category: 'featured' | 'cloud' | 'work' | 'practical'
}

export interface SanityAchievement {
  _id: string
  title: string
  icon?: string | null
  description?: string | null
  date?: string | null
}

export interface SanitySkillCategory {
  _id: string
  category: string
  items: string[]
}

export interface SanityResumeFileCard {
  label: string
  url: string
}

export interface SanityEducationEntry {
  institution: string
  period: string
  description?: string
}

export interface SanityExperienceEntry {
  title: string
  organization: string
  period: string
  current?: boolean
  highlights?: string[]
}

export interface SanityResumeProjectEntry {
  title: string
  subtitle?: string
  highlights?: string[]
}

export interface SanityTrainingEntry {
  title: string
  organization: string
  period: string
  description?: string
}

export interface SanityResume {
  education: SanityEducationEntry[]
  experience: SanityExperienceEntry[]
  projects: SanityResumeProjectEntry[]
  training: SanityTrainingEntry[]
}