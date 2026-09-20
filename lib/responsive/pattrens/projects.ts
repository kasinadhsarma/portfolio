export const projectsPagePatterns = {
  container: "container mx-auto space-y-6",
}

export const projectsHeaderPatterns = {
  wrapper: "flex flex-col gap-4 md:flex-row md:items-center md:justify-between",
  inner: "flex flex-col gap-2",
  titleRow: "flex items-center gap-4",
  title: "text-4xl font-bold",
  metaRow: "flex items-center gap-4",
  description: "text-muted-foreground",
  badgeLink: "hover:opacity-80 transition-opacity",
  divider: "h-1 w-16 bg-primary mt-2",
}

export const projectsClientPatterns = {
  tabsScrollWrapper:
    "w-full overflow-x-auto [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden",
  tabsList: "inline-flex w-max md:w-auto",
  tabsTrigger: "flex items-center gap-2 whitespace-nowrap",
  tabsContent: "mt-6",
  emptyState: "text-center py-12 text-muted-foreground",
  grid: "grid gap-6 sm:grid-cols-2 lg:grid-cols-3",
}

export const projectCardPatterns = {
  card: "overflow-hidden flex flex-col",
  imageWrapper: "aspect-video overflow-hidden",
  image: "w-full h-full object-cover transition-transform hover:scale-105",
  headerRow: "flex items-start justify-between gap-2",
  title: "text-lg line-clamp-2",
  featuredBadge: "text-xs shrink-0",
  content: "flex-grow space-y-4",
  description: "line-clamp-3",
  techRow: "flex flex-wrap gap-2",
  techBadge: "capitalize",
  categoryBadge: "bg-primary/5",
  footer: "border-t bg-muted/50 pt-4",
  footerRow: "flex w-full justify-between",
  footerLink: "flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors",
}
