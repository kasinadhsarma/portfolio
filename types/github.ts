export interface GitHubRepo {
  id: number
  name: string
  full_name?: string
  description?: string | null
  html_url: string
  homepage?: string | null
  topics?: string[]
  language?: string | null
  stargazers_count: number
  forks_count?: number
  updated_at?: string
}

export interface GitHubApiError {
  error: string
  status?: number
}
