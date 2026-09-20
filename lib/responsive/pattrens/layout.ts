export const mainNavPatterns = {
  root: "fixed left-0 right-0 z-50 flex justify-center p-4 pointer-events-none bottom-0",
  innerRow: "flex items-center gap-3 pointer-events-auto",
  mobilePill: "flex md:hidden items-center gap-1 bg-background/95 p-1.5 rounded-full shadow-lg backdrop-blur-sm border border-border/50",
  mobileMenuButton: "text-primary rounded-full hover:bg-primary/10 transition-all duration-200",
  mobileThemeButton: "text-primary rounded-full transition-all duration-200 hover:text-primary/90 hover:bg-primary/10 active:scale-95",
  desktopNav:
    "hidden md:flex md:gap-8 bg-background/95 px-8 py-3 rounded-full shadow-lg items-center backdrop-blur-sm border border-border/50 transition-all duration-300 ease-in-out hover:shadow-xl hover:bg-background/98 hover:border-primary/20 hover:scale-[1.02]",
  desktopLink: "text-sm font-medium text-muted-foreground transition-all duration-200 hover:text-foreground hover:scale-105 active:scale-95",
  desktopLinkActive: "text-foreground font-bold",
  externalIcon: "ml-1 inline-block h-3 w-3",
  desktopThemeButton: "text-primary shrink-0 transition-all duration-200 hover:text-primary/90 hover:bg-primary/10 active:scale-95",
  sheetContent: "w-full bg-card/95 border-none rounded-t-3xl transform transition-all duration-500 ease-in-out",
  sheetContentSlideUp: "animate-slide-up",
  sheetContentSlideDown: "animate-slide-down",
  mobileNav:
    "flex flex-col gap-2 transform transition-all duration-300 data-[state=open]:translate-y-0 data-[state=open]:opacity-100 data-[state=closed]:translate-y-4 data-[state=closed]:opacity-0",
  mobileLink: "text-lg font-medium px-4 py-3 text-primary transition-all duration-200 rounded-lg hover:text-primary hover:bg-primary/10 active:scale-95",
  mobileLinkActive: "text-primary/90 bg-primary/20",
  mobileExternalIcon: "ml-2 inline-block h-4 w-4",
  mobileResumeWrapper: "px-4 py-2",
  scrollTopButton: "fixed bottom-20 right-4 text-primary pointer-events-auto transition-all duration-200 hover:text-primary/90 hover:bg-primary/10 md:bottom-24",
  scrollTopButtonHidden: "opacity-0 pointer-events-none",
}
