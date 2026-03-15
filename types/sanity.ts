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
  category: 'ai' | 'web' | 'cybersecurity' | 'database' | 'cloud' | 'mobile' | 'desktop' | 'other'
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
  category: string
  github?: string | null
  liveUrl?: string | null
  featured: boolean
  publishedAt: string
}