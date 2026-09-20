import Link from "next/link"
import { Button } from "@/components/ui/button"
import { notFound } from "@/lib/responsive/pattrens/not-found"
export default function NotFound() {
  return (
    <div className={notFound.container}>
      <h1 className={notFound.h1}>
        404
      </h1>
      <div className={notFound.divider}></div>
      <h2 className={notFound.h2}>Page not found</h2>
      <p className={notFound.p}>
        The page you're looking for doesn't exist or may have been moved.
      </p>
      <Button asChild>
        <Link href="/">Back to home</Link>
      </Button>
    </div>
  )
}
