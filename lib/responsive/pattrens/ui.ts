export const badgePatterns = {
  base: "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  variantDefault: "border-transparent bg-primary text-primary-foreground hover:bg-primary/80",
  variantSecondary: "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
  variantDestructive: "border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80",
  variantOutline: "text-foreground",
}

export const buttonPatterns = {
  base: "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  variantDefault: "bg-primary text-primary-foreground hover:bg-primary/90",
  variantDestructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
  variantOutline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
  variantSecondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
  variantGhost: "hover:bg-accent hover:text-accent-foreground",
  variantLink: "text-primary underline-offset-4 hover:underline",
  variantGradient:
    "bg-gradient-to-r from-primary to-primary/70 text-primary-foreground hover:from-primary/90 hover:to-primary/60",
  sizeDefault: "h-10 px-4 py-2",
  sizeSm: "h-9 rounded-md px-3",
  sizeLg: "h-11 rounded-md px-8",
  sizeIcon: "h-10 w-10",
}

export const cardPatterns = {
  root: "rounded-lg border bg-card text-card-foreground shadow-sm card-hover glass",
  header: "flex flex-col space-y-1.5 p-6",
  title: "text-2xl font-semibold leading-none tracking-tight gradient-text",
  description: "text-sm text-muted-foreground",
  content: "p-6 pt-0",
  footer: "flex items-center p-6 pt-0",
}

export const dropdownMenuPatterns = {
  subTrigger:
    "flex cursor-default gap-2 select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none focus:bg-accent data-[state=open]:bg-accent [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  subTriggerInset: "pl-8",
  subContent:
    "z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-lg data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
  content:
    "z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
  item: "relative flex cursor-default select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none transition-colors focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  itemInset: "pl-8",
  checkboxItem:
    "relative flex cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none transition-colors focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
  checkboxItemIndicatorWrapper: "absolute left-2 flex h-3.5 w-3.5 items-center justify-center",
  radioItem:
    "relative flex cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none transition-colors focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
  radioItemIndicatorWrapper: "absolute left-2 flex h-3.5 w-3.5 items-center justify-center",
  label: "px-2 py-1.5 text-sm font-semibold",
  labelInset: "pl-8",
  separator: "-mx-1 my-1 h-px bg-muted",
  shortcut: "ml-auto text-xs tracking-widest opacity-60",
}

export const sheetPatterns = {
  overlay:
    "fixed inset-0 z-50 bg-black/80 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
  base: "fixed z-50 gap-4 bg-background p-6 shadow-lg transition ease-in-out data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:duration-300 data-[state=open]:duration-500",
  sideTop: "inset-x-0 top-0 data-[state=closed]:slide-out-to-top data-[state=open]:slide-in-from-top",
  sideBottom: "inset-x-0 bottom-0 data-[state=closed]:slide-out-to-bottom data-[state=open]:slide-in-from-bottom",
  sideLeft:
    "inset-y-0 left-0 h-full w-3/4 data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left sm:max-w-sm",
  sideRight:
    "inset-y-0 right-0 h-full w-3/4 data-[state=closed]:slide-out-to-right data-[state=open]:slide-in-from-right sm:max-w-sm",
  closeButton:
    "absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-secondary",
  closeIcon: "h-4 w-4 text-amber-400",
  header: "flex flex-col space-y-2 text-center sm:text-left",
  footer: "flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2",
  title: "text-lg font-semibold text-foreground",
  description: "text-sm text-muted-foreground",
}

export const tabsPatterns = {
  list: "inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground",
  trigger:
    "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm",
  content: "mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
}

export const toastPatterns = {
  viewport:
    "fixed top-0 z-[100] flex max-h-screen w-full flex-col-reverse p-4 sm:bottom-0 sm:right-0 sm:top-auto sm:flex-col md:max-w-[420px]",
  base: "group pointer-events-auto relative flex w-full items-center justify-between space-x-4 overflow-hidden rounded-md border p-6 pr-8 shadow-lg transition-all data-[swipe=cancel]:translate-x-0 data-[swipe=end]:translate-x-[var(--radix-toast-swipe-end-x)] data-[swipe=move]:translate-x-[var(--radix-toast-swipe-move-x)] data-[swipe=move]:transition-none data-[state=open]:animate-in data-[state=closed]:animate-out data-[swipe=end]:animate-out data-[state=closed]:fade-out-80 data-[state=closed]:slide-out-to-right-full data-[state=open]:slide-in-from-top-full data-[state=open]:sm:slide-in-from-bottom-full",
  variantDefault: "border bg-background text-foreground",
  variantDestructive: "destructive group border-destructive bg-destructive text-destructive-foreground",
  action:
    "inline-flex h-8 shrink-0 items-center justify-center rounded-md border bg-transparent px-3 text-sm font-medium ring-offset-background transition-colors hover:bg-secondary focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 group-[.destructive]:border-muted/40 group-[.destructive]:hover:border-destructive/30 group-[.destructive]:hover:bg-destructive group-[.destructive]:hover:text-destructive-foreground group-[.destructive]:focus:ring-destructive",
  close:
    "absolute right-2 top-2 rounded-md p-1 text-foreground/50 opacity-0 transition-opacity hover:text-foreground focus:opacity-100 focus:outline-none focus:ring-2 group-hover:opacity-100 group-[.destructive]:text-red-300 group-[.destructive]:hover:text-red-50 group-[.destructive]:focus:ring-red-400 group-[.destructive]:focus:ring-offset-red-600",
  title: "text-sm font-semibold",
  description: "text-sm opacity-90",
}

export const toasterPatterns = {
  contentWrapper: "grid gap-1",
}

export const scrollSectionPatterns = {
  base: "transform transition-all duration-700 ease-out",
  visible: "translate-y-0 translate-x-0 opacity-100",
  hidden: "opacity-0",
  directionUp: "translate-y-8",
  directionDown: "-translate-y-8",
  directionLeft: "translate-x-8",
  directionRight: "-translate-x-8",
}

export const themeTogglePatterns = {
  track: "relative flex h-8 w-16 cursor-pointer items-center rounded-full p-1 transition-colors duration-300",
  trackDark: "bg-slate-700",
  trackLight: "bg-amber-100",
  thumb: "absolute h-6 w-6 rounded-full transition-transform duration-300",
  thumbDark: "translate-x-8 bg-slate-900",
  thumbLight: "translate-x-0 bg-amber-400",
  sunIcon: "absolute left-1.5 h-4 w-4 text-amber-400 transition-opacity",
  moonIcon: "absolute right-1.5 h-4 w-4 text-slate-200 transition-opacity",
  iconVisible: "opacity-100",
  iconHidden: "opacity-0",
}

export const animatedSectionPatterns = {
  base: "transform transition-all duration-700 ease-out",
  visible: "translate-y-0 opacity-100",
  hidden: "translate-y-8 opacity-0",
}

export const typingEffectPatterns = {
  text: "text-xl font-medium bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent dark:from-primary/90 dark:via-primary/70 dark:to-primary/50",
  cursor: "animate-blink text-primary",
}

export const certificatesCarouselPatterns = {
  root: "relative w-full group",
  scrollRow: "flex gap-6 overflow-x-auto no-scrollbar scroll-smooth",
  item: "flex-none w-48 h-48 relative rounded-xl border border-border/40 bg-card p-4 shadow-sm transition-transform hover:scale-105 hover:shadow-md",
  itemClickable: "cursor-pointer",
  itemStatic: "cursor-default",
  image: "object-contain p-2",
  navLeft: "absolute left-0 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity",
  navRight: "absolute right-0 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity",
}

export const resumeDropdownPatterns = {
  triggerButton: "gap-2",
  content: "w-56",
  link: "flex items-center gap-2 cursor-pointer w-full",
}
