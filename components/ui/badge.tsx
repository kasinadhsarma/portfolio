import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"
import { badgePatterns as p } from "@/lib/responsive/pattrens/ui"

const badgeVariants = cva(
  p.base,
  {
    variants: {
      variant: {
        default: p.variantDefault,
        secondary: p.variantSecondary,
        destructive: p.variantDestructive,
        outline: p.variantOutline,
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
