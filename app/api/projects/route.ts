import { NextResponse } from 'next/server'
import { GitHubRepo } from '@/types/github'

const GITHUB_TOKEN = process.env.GITHUB_TOKEN

const REPOS = [
  'vicharcha/Web',
  'kasinadhsarma/urbandevlopment',
  'VishwamAI/ProtienFlex',
  'VishwamAI/VishwamAI',
  'VishwamAI/NeuroFlex',
  'VishwamAI/jobcity',
  'Exploit0xfffff/Intelrepo',
  'VishwamAI/PikasuBirdAi',
  'kasinadhsarma/Sai-Krishna-Home-Care',
]

export async function GET() {
  if (!GITHUB_TOKEN) {
    console.error('GitHub token not found in environment variables')
    return NextResponse.json(
      { error: 'GitHub token not configured' },
      { status: 500 }
    )
  }

  try {
    const repos = await Promise.all(
      REPOS.map(async (repo) => {
        try {
          const response = await fetch(`https://api.github.com/repos/${repo}`, {
            headers: {
              Authorization: `token ${GITHUB_TOKEN}`,
              Accept: 'application/vnd.github.v3+json',
            },
            next: { revalidate: 3600 } // Cache for 1 hour
          })

          if (!response.ok) {
            console.error(`Failed to fetch ${repo}:`, await response.text())
            return null
          }

          const data = await response.json()
          return {
            id: data.id,
            name: data.name,
            full_name: data.full_name,
            description: data.description || null,
            html_url: data.html_url,
            homepage: data.homepage || null,
            topics: data.topics || [],
            language: data.language || null,
            stargazers_count: data.stargazers_count || 0,
            forks_count: data.forks_count,
            updated_at: data.updated_at
          } as GitHubRepo
        } catch (error) {
          console.error(`Error fetching ${repo}:`, error)
          return null
        }
      })
    )

    // Filter out any failed requests
    const validRepos = repos.filter((repo): repo is GitHubRepo => repo !== null)

    return NextResponse.json(validRepos)
  } catch (error) {
    console.error('Failed to fetch projects:', error)
    return NextResponse.json(
      { error: 'Failed to fetch projects' },
      { status: 500 }
    )
  }
}
