import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"
import { buttonPatterns as p } from "@/lib/responsive/pattrens/ui"

const buttonVariants = cva(
  p.base,
  {
    variants: {
      variant: {
        default: p.variantDefault,
        destructive: p.variantDestructive,
        outline: p.variantOutline,
        secondary: p.variantSecondary,
        ghost: p.variantGhost,
        link: p.variantLink,
        gradient: p.variantGradient,
      },
      size: {
        default: p.sizeDefault,
        sm: p.sizeSm,
        lg: p.sizeLg,
        icon: p.sizeIcon,
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
  },
)
Button.displayName = "Button"

export { Button, buttonVariants }
