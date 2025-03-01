'use client'

import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"

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
      <Card className="h-full bg-card/50 backdrop-blur-sm border-2 hover:border-primary/50">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-xl font-bold">{title}</CardTitle>
            <div
              className={cn(
                "px-3 py-1 text-sm font-medium rounded-full",
                status === "Ongoing" && "bg-yellow-500/10 text-yellow-500",
                status === "Active" && "bg-green-500/10 text-green-500",
                status === "Completed" && "bg-blue-500/10 text-blue-500",
              )}
            >
              {status}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground mb-4 text-base">{description}</p>
          <div className="flex flex-wrap gap-2">
            {technologies.map((tech) => (
              <div key={tech} className="px-3 py-1 text-sm rounded-full bg-primary/10 text-primary font-medium">
                {tech}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}

