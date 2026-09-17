import Link from "next/link"
import { Button } from "@/components/ui/button"

export default function NotFound() {
  return (
    <div className="container mx-auto px-4 py-24 flex flex-col items-center text-center space-y-6">
      <h1 className="text-6xl font-bold bg-gradient-to-r from-primary to-primary/50 bg-clip-text text-transparent">
        404
      </h1>
      <div className="h-1 w-20 bg-gradient-to-r from-primary to-primary/50"></div>
      <h2 className="text-2xl font-semibold">Page not found</h2>
      <p className="text-muted-foreground max-w-md">
        The page you're looking for doesn't exist or may have been moved.
      </p>
      <Button asChild>
        <Link href="/">Back to home</Link>
      </Button>
    </div>
  )
}
