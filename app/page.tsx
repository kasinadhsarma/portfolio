"use client"

import * as React from "react"
import { useState } from "react"
import { TypeAnimation } from "react-type-animation"
import Image from "next/image"
import { Button } from "../components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs"
import { Github, Linkedin, Mail, Twitter } from "lucide-react"

export default function EnhancedDeveloperPortfolio() {
  // State and data declarations (existing code)
  const [activeTab, setActiveTab] = useState("about")

  const skills = [
    { name: "React/Next.js", level: 90 },
    { name: "TypeScript", level: 85 },
    { name: "Node.js", level: 80 },
    { name: "Python", level: 85 },
    { name: "DevOps/CI/CD", level: 75 },
    { name: "Security", level: 80 }
  ]

  const projects = [
    {
      name: "Cloud Automation",
      category: "DevOps",
      image: "/projects/cloud-automation.jpg"
    },
    {
      name: "E-commerce Platform",
      category: "Full Stack",
      image: "/projects/ecommerce.jpg"
    },
    {
      name: "Threat Detection System",
      category: "Security",
      image: "/projects/threat-detection.jpg"
    }
  ]

  const experience = [
    {
      title: "EDZU - Internship",
      period: "December 2023 - March 2024",
      description: "Successfully executed penetration testing and participated in development tasks, enhancing expertise in security technology."
    },
    {
      title: "Machine Learning Training",
      period: "September 2023 - November 2023",
      description: "Completed comprehensive machine learning training program. Learned about various ML projects and their implementation."
    }
  ]

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-4 md:p-8">
      {/* Animated Background - existing code */}
      <div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute -inset-[10px] opacity-50">
          <div className="absolute mix-blend-multiply filter blur-xl top-0 -left-4 w-96 h-96 bg-purple-500 rounded-full animate-blob"></div>
          <div className="absolute mix-blend-multiply filter blur-xl top-0 -right-4 w-96 h-96 bg-yellow-500 rounded-full animate-blob animation-delay-2000"></div>
          <div className="absolute mix-blend-multiply filter blur-xl -bottom-8 left-20 w-96 h-96 bg-pink-500 rounded-full animate-blob animation-delay-4000"></div>
        </div>
      </div>

      {/* Header and Social Links - existing code */}
      <header className="text-center mb-12 relative z-10">
        <h1 className="text-4xl md:text-6xl font-bold mb-4 glitch" data-text="Kasinadh Sarma">
          Kasinadh Sarma
        </h1>
        <div className="h-8 md:h-12">
          <TypeAnimation
            sequence={["Full Stack Developer", 2000, "DevOps Engineer", 2000, "Security Enthusiast", 2000]}
            wrapper="span"
            speed={50}
            style={{ fontSize: "1.25em", display: "inline-block" }}
            repeat={Infinity}
          />
        </div>
      </header>
      {/* Social Links */}
      <div className="flex justify-center gap-4 mb-8">
        <Button variant="ghost" size="icon" className="hover:text-purple-400 transition-colors">
          <Github className="h-5 w-5" />
        </Button>
        <Button variant="ghost" size="icon" className="hover:text-blue-400 transition-colors">
          <Linkedin className="h-5 w-5" />
        </Button>
        <Button variant="ghost" size="icon" className="hover:text-green-400 transition-colors">
          <Mail className="h-5 w-5" />
        </Button>
        <Button variant="ghost" size="icon" className="hover:text-pink-400 transition-colors">
          <Twitter className="h-5 w-5" />
        </Button>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="about" className="max-w-4xl mx-auto">
        <TabsList className="grid w-full grid-cols-4 mb-8">
          <TabsTrigger value="about">About</TabsTrigger>
          <TabsTrigger value="experience">Experience</TabsTrigger>
          <TabsTrigger value="skills">Skills</TabsTrigger>
          <TabsTrigger value="projects">Projects</TabsTrigger>
        </TabsList>

        <TabsContent value="about">
          <Card className="bg-gray-800 border-purple-500 hover:shadow-lg hover:shadow-purple-500/50 transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-purple-400">About Me</CardTitle>
            </CardHeader>
            <CardContent className="text-gray-300">
              <p>Full Stack Developer and DevOps enthusiast with a passion for building secure, scalable applications. Experienced in modern web technologies and cloud infrastructure.</p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="experience">
          <Card className="bg-gray-800 border-yellow-500 hover:shadow-lg hover:shadow-yellow-500/50 transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-yellow-400">Work Experience</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {experience.map((job, index) => (
                <div key={index} className="relative pl-8 pb-8 group">
                  <div className="absolute left-0 top-0 w-2 h-full bg-yellow-500 group-hover:bg-green-500 transition-colors duration-300"></div>
                  <div className="absolute left-0 top-0 w-6 h-6 bg-yellow-500 rounded-full border-4 border-gray-800 group-hover:bg-green-500 transition-colors duration-300"></div>
                  <h3 className="text-lg font-semibold text-yellow-400 group-hover:text-green-400 transition-colors duration-300">{job.title}</h3>
                  <p className="text-sm text-gray-400">{job.period}</p>
                  <p className="mt-2 text-gray-300">{job.description}</p>
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="skills">
          <Card className="bg-gray-800 border-red-500 hover:shadow-lg hover:shadow-red-500/50 transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-red-400">Technical Skills</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {skills.map((skill, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-300">{skill.name}</span>
                    <span className="text-gray-400">{skill.level}%</span>
                  </div>
                  <div className="relative pt-1">
                    <div className="overflow-hidden h-2 text-xs flex rounded bg-gray-700">
                      <div
                        style={{ width: `${skill.level}%` }}
                        className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-gradient-to-r from-red-500 to-yellow-500 transition-all duration-300"
                      ></div>
                    </div>
                    <div
                      style={{ left: `${skill.level}%` }}
                      className="absolute -top-1 w-4 h-4 rounded-full bg-white border-2 border-red-500 transition-all duration-300"
                    ></div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="projects">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projects.map((project, index) => (
              <Card key={index} className="bg-gray-800 border-blue-500 hover:shadow-lg hover:shadow-blue-500/50 transition-all duration-300 group overflow-hidden">
                <div className="relative aspect-video overflow-hidden">
                  <Image
                    src={project.image}
                    alt={project.name}
                    fill
                    className="object-cover transition-transform duration-300 group-hover:scale-110"
                  />
                  <div className="absolute inset-0 bg-gradient-to-br from-blue-500/75 to-purple-500/75 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                    <Button variant="secondary" size="sm" className="bg-white text-gray-900 hover:bg-gray-200">
                      View Project
                    </Button>
                  </div>
                </div>
                <CardContent className="pt-4">
                  <h3 className="text-lg font-semibold text-blue-400 group-hover:text-purple-400 transition-colors duration-300">{project.name}</h3>
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
