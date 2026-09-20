'use client'

import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import { researchCardPatterns as p } from "@/lib/responsive/pattrens/research"

interface ResearchCardProps {
  title: string
  status: string
  description: string
  technologies: string[]
}

export function ResearchCard({
  title,
  status,
  description,
  technologies,
}: ResearchCardProps) {
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.2 }}
    >
      <Card className={p.card}>
        <CardHeader>
          <div className={p.headerRow}>
            <CardTitle className={p.title}>{title}</CardTitle>
            <div
              className={cn(
                p.statusBadge,
                status === "Ongoing" && p.statusOngoing,
                status === "Active" && p.statusActive,
                status === "Completed" && p.statusCompleted,
              )}
            >
              {status}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <p className={p.description}>{description}</p>
          <div className={p.techRow}>
            {technologies.map((tech) => (
              <div key={tech} className={p.techBadge}>
                {tech}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
