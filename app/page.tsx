"use client"

import React, { useEffect } from 'react'
import { TypeAnimation } from 'react-type-animation'
import { Button } from "../components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs"
import { Progress } from "../components/ui/progress"
import { GithubIcon, LinkedinIcon, MailIcon, TwitterIcon } from "lucide-react"
import Image from 'next/image'
import { useState } from 'react'

export default function EnhancedDeveloperPortfolio() {
  const [activeTab, setActiveTab] = useState('about')

  // Data arrays
  const skills = [
    { name: "React/Next.js", level: 90 },
    { name: "TypeScript", level: 85 },
    { name: "Node.js", level: 85 },
    { name: "Python", level: 80 },
    { name: "DevOps", level: 75 },
    { name: "Security", level: 70 }
  ]

  const projects = [
    {
      name: "Project 1",
      category: "Web Development",
      description: "Modern web application built with React and Next.js",
      image: "/projects/project1.jpg"
    },
    {
      name: "Project 2",
      category: "DevOps",
      description: "CI/CD pipeline implementation with GitHub Actions",
      image: "/projects/project2.jpg"
    },
    {
      name: "Project 3",
      category: "Security",
      description: "Security analysis and penetration testing tools",
      image: "/projects/project3.jpg"
    }
  ]

  const experience = [
    {
      title: "Senior Developer",
      period: "2022 - Present",
      description: "Leading development of enterprise applications using React and Node.js"
    },
    {
      title: "DevOps Engineer",
      period: "2020 - 2022",
      description: "Implemented CI/CD pipelines and managed cloud infrastructure"
    },
    {
      title: "Security Researcher",
      period: "2018 - 2020",
      description: "Conducted security audits and implemented security measures"
    }
  ]

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 responsive-padding">
      {/* Animated Background */}
      <div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute mix-blend-screen top-0 -left-4 w-96 h-96 bg-purple-500/80 rounded-full animate-blob"></div>
          <div className="absolute mix-blend-screen top-0 -right-4 w-96 h-96 bg-yellow-500/80 rounded-full animate-blob animation-delay-2000"></div>
          <div className="absolute mix-blend-screen -bottom-8 left-20 w-96 h-96 bg-pink-500/80 rounded-full animate-blob animation-delay-4000"></div>
        </div>
      </div>
      {/* Header */}
      <header className="mb-12 relative z-10">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <h1 className="responsive-heading glitch" data-text="Kasinadh Sarma">
            Kasinadh Sarma
          </h1>
          <div className="typing-container">
            <TypeAnimation
              sequence={[
                'Full Stack Developer',
                1500,
                'DevOps Engineer',
                1500,
                'Security Researcher',
                1500
              ]}
              wrapper="span"
              speed={50}
              repeat={Infinity}
              cursor={true}
              className="text-xl md:text-2xl text-yellow-400"
            />
          </div>
        </div>
      </header>

      {/* Social Links */}
      <div className="flex justify-center gap-4 mb-8">
        <Button variant="ghost" size="icon" className="hover:text-purple-400 transition-colors">
          <GithubIcon size={20} />
        </Button>
        <Button variant="ghost" size="icon" className="hover:text-blue-400 transition-colors">
          <LinkedinIcon size={20} />
        </Button>
        <Button variant="ghost" size="icon" className="hover:text-green-400 transition-colors">
          <MailIcon size={20} />
        </Button>
        <Button variant="ghost" size="icon" className="hover:text-cyan-400 transition-colors">
          <TwitterIcon size={20} />
        </Button>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="about" onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-4 bg-gray-800 rounded-lg p-1">
          <TabsTrigger value="about" className="data-[state=active]:bg-gray-700">About</TabsTrigger>
          <TabsTrigger value="experience" className="data-[state=active]:bg-gray-700">Experience</TabsTrigger>
          <TabsTrigger value="skills" className="data-[state=active]:bg-gray-700">Skills</TabsTrigger>
          <TabsTrigger value="projects" className="data-[state=active]:bg-gray-700">Projects</TabsTrigger>
        </TabsList>

        <TabsContent value="about" className="space-y-4">
          <Card className="bg-gray-800 border-green-500 hover:shadow-lg hover:shadow-green-500/50 transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-green-400">About Me</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-300">
                Full Stack Developer and DevOps enthusiast with a passion for building secure, scalable applications.
                Experienced in modern web technologies and cloud infrastructure.
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="experience" className="space-y-4">
          {experience.map((job, index) => (
            <div key={index} className="relative pl-8 pb-8 group">
              <div className="absolute left-0 top-0 w-2 h-full bg-yellow-500 group-hover:bg-green-500 transition-colors duration-300"></div>
              <div className="absolute left-0 top-0 w-6 h-6 bg-yellow-500 rounded-full border-4 border-gray-800 group-hover:bg-green-500 transition-colors duration-300"></div>
              <h3 className="text-lg font-semibold text-yellow-400 group-hover:text-green-400 transition-colors duration-300">{job.title}</h3>
              <p className="text-sm text-gray-400">{job.period}</p>
              <p className="mt-2 text-gray-300">{job.description}</p>
            </div>
          ))}
        </TabsContent>

        <TabsContent value="skills" className="space-y-4">
          <Card className="bg-gray-800 border-red-500 hover:shadow-lg hover:shadow-red-500/50 transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-red-400">Technical Skills</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {skills.map((skill, index) => (
                <div key={index} className="space-y-2 group">
                  <div className="flex justify-between">
                    <span className="text-gray-300 group-hover:text-red-400 transition-colors duration-300">{skill.name}</span>
                    <span className="text-gray-400">{skill.level}%</span>
                  </div>
                  <Progress value={skill.level} className="h-2 overflow-hidden rounded-full bg-gray-700" indicatorClassName="bg-gradient-to-r from-red-500 via-yellow-500 to-orange-500 transition-all duration-500 ease-in-out" />
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="projects" className="mt-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projects.map((project, index) => (
              <Card key={index} className="bg-gray-800 border-blue-500 hover:shadow-lg hover:shadow-blue-500/50 transition-all duration-300 group">
                <div className="relative overflow-hidden aspect-video">
                  <Image
                    src={project.image}
                    alt={project.name}
                    width={400}
                    height={300}
                    className="object-cover rounded-t-lg transition-transform duration-500 group-hover:scale-110"
                  />
                  <div className="absolute inset-0 bg-gradient-to-br from-blue-500/80 to-purple-500/80 opacity-0 group-hover:opacity-100 transition-all duration-500 flex flex-col items-center justify-center gap-4 p-6">
                    <h4 className="text-white text-lg font-semibold text-center opacity-0 group-hover:opacity-100 transform -translate-y-4 group-hover:translate-y-0 transition-all duration-500">{project.description}</h4>
                    <Button variant="secondary" size="sm" className="bg-white text-gray-900 hover:bg-gray-200 opacity-0 group-hover:opacity-100 transform translate-y-4 group-hover:translate-y-0 transition-all duration-500">
                      View Project
                    </Button>
                  </div>
                </div>
                <CardContent className="pt-4">
                  <h3 className="text-lg font-semibold text-blue-400 group-hover:text-green-400 transition-colors duration-300">{project.name}</h3>
                  <p className="text-sm text-gray-400">{project.category}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
