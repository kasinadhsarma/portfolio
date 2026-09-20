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