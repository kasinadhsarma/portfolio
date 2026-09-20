export const researchContentPatterns = {
  container: "container mx-auto px-4 py-12 space-y-12",
  title: "text-4xl font-bold bg-gradient-to-r from-primary to-primary/50 bg-clip-text text-transparent",
  divider: "h-1 w-20 bg-gradient-to-r from-primary to-primary/50 mt-2",
  sectionHeading: "text-2xl font-semibold mb-6",
  grid: "grid gap-6",
}

export const publicationCardPatterns = {
  card: "hover:bg-accent/50 transition-colors cursor-pointer",
  headerRow: "flex items-center justify-between",
  title: "text-lg hover:text-primary transition-colors",
}

export const researchCardPatterns = {
  card: "h-full bg-card/50 backdrop-blur-sm border-2 hover:border-primary/50",
  headerRow: "flex items-center justify-between",
  title: "text-xl font-bold",
  statusBadge: "px-3 py-1 text-sm font-medium rounded-full",
  statusOngoing: "bg-yellow-500/10 text-yellow-500",
  statusActive: "bg-green-500/10 text-green-500",
  statusCompleted: "bg-blue-500/10 text-blue-500",
  description: "text-muted-foreground mb-4 text-base",
  techRow: "flex flex-wrap gap-2",
  techBadge: "px-3 py-1 text-sm rounded-full bg-primary/10 text-primary font-medium",
}
