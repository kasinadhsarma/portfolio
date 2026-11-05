// Server component – fetches all home page data from SethuCMS at request time
// and passes it as props to the client shell to avoid waterfalling client fetches.
import { getHomeContent, getAboutContent, getActivities, getSkills, getCertificates, getResumeFiles } from '@/lib/content'
import HomePageClient from '@/components/pages/home/page-client'

export default async function HomePage() {
  const [hero, about, activities, skills, certificates, resumeFiles] = await Promise.all([
    getHomeContent(),
    getAboutContent(),
    getActivities(),
    getSkills(),
    getCertificates(),
    getResumeFiles(),
  ])

  return (
    <HomePageClient
      hero={hero}
      about={about}
      activities={activities}
      skills={skills}
      certificates={certificates}
      resumeFiles={resumeFiles}
    />
  )
}
