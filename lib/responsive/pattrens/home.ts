export const homePagePatterns = {
  root: "relative",
  backgroundWrapper: "fixed inset-0 z-0",
  backgroundGradient:
    "absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-accent/5 via-background to-background",
  contentWrapper: "relative z-10",
}

export const heroPatterns = {
  section: "relative py-20",
  container: "container max-w-6xl 2xl:max-w-[1600px]",
  layout: "flex flex-col md:flex-row items-center gap-12",
  avatarWrapper: "flex-shrink-0",
  avatarFrame: "relative w-48 h-48 md:w-64 md:h-64",
  avatarGlow:
    "absolute inset-0 bg-gradient-to-r from-accent/20 to-accent/30 rounded-full blur-2xl opacity-50 animate-pulse",
  avatarImage: "rounded-full object-cover border-4 border-accent/20",
  contentColumn: "flex-1 text-center md:text-left",
  heading: "text-4xl md:text-5xl font-bold mb-6 text-foreground",
  typingWrapper: "h-10 mb-8",
  tagline: "text-muted-foreground mb-8 max-w-2xl",
  ctaRow: "flex flex-wrap gap-4 justify-center md:justify-start",
  ctaButton: "gap-2",
  socialRow: "flex gap-4 mt-8 justify-center md:justify-start",
  socialLink: "text-muted-foreground hover:text-foreground transition-colors",
}

export const aboutPatterns = {
  section: "py-16",
  container: "container max-w-6xl 2xl:max-w-[1600px]",
  heading:
    "text-3xl font-bold mb-8 bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent",
  proseWrapper: "prose dark:prose-invert max-w-none",
  paragraph: "mb-4 text-foreground dark:text-foreground/90",
}

export const whatImDoingPatterns = {
  section: "py-16 bg-gradient-to-b from-background to-accent/5",
  container: "container max-w-6xl 2xl:max-w-[1600px]",
  heading:
    "text-3xl font-bold mb-8 bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent",
  grid: "grid md:grid-cols-2 lg:grid-cols-4 gap-6",
  card:
    "p-6 group transition-all duration-300 hover:shadow-lg hover:shadow-primary/10 hover:scale-[1.02] dark:bg-card/80 backdrop-blur-sm border-primary/20 hover:border-primary/40 bg-gradient-to-br from-card via-card/95 to-card/90",
  cardInner: "flex flex-col items-center text-center space-y-4",
  iconWrapper:
    "w-16 h-16 rounded-xl bg-primary/10 flex items-center justify-center group-hover:scale-110 transition-transform duration-300 group-hover:bg-primary/20",
  icon: "w-8 h-8 group-hover:scale-110 transition-transform duration-300",
  textWrapper: "space-y-2",
  title: "font-semibold text-foreground group-hover:text-primary transition-colors",
  description: "text-sm text-muted-foreground group-hover:text-muted-foreground/80",
}

export const skillsPatterns = {
  section: "py-16 bg-gradient-to-b from-accent/5 to-background",
  container: "container max-w-6xl 2xl:max-w-[1600px]",
  heading:
    "text-3xl font-bold mb-8 bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent",
  grid: "grid md:grid-cols-2 lg:grid-cols-4 gap-6",
  card:
    "p-6 transition-all duration-300 hover:shadow-lg hover:shadow-primary/10 hover:scale-[1.02] dark:bg-card/80 backdrop-blur-sm border-primary/20 hover:border-primary/40 bg-gradient-to-br from-card via-card/95 to-card/90",
  categoryTitle: "font-semibold text-primary mb-4",
  list: "space-y-2",
  listItem: "text-muted-foreground hover:text-foreground transition-colors flex items-center gap-2",
  bullet: "w-1.5 h-1.5 rounded-full bg-primary/60",
}

export const certificatesPatterns = {
  section: "py-16 bg-gradient-to-b from-background to-accent/5",
  container: "container max-w-6xl 2xl:max-w-[1600px] space-y-16",
  heading:
    "text-3xl font-bold mb-8 bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent",
}

export const contactCtaPatterns = {
  section: "py-16",
  container: "container max-w-6xl 2xl:max-w-[1600px]",
  card:
    "relative p-12 rounded-3xl overflow-hidden text-center bg-gradient-to-br from-card via-card/95 to-card/90 dark:from-card/90 dark:to-card/70 border border-primary/20 hover:border-primary/40 shadow-lg hover:shadow-primary/10",
  content: "relative z-10 max-w-3xl mx-auto",
  heading:
    "text-3xl font-bold mb-6 bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent",
  description: "text-muted-foreground dark:text-muted-foreground/80 mb-8",
  button: "px-8 bg-primary hover:bg-primary/90 text-primary-foreground transition-all duration-300 hover:scale-105",
  glow:
    "absolute inset-0 bg-gradient-to-r from-primary/10 via-primary/5 to-primary/10 dark:from-primary/5 dark:via-primary/2 dark:to-primary/5 opacity-50",
}
