"use client";

import { usePathname, useSearchParams } from "next/navigation";
import { useEffect } from "react";

declare global {
  interface Window {
    gtag?: (...args: unknown[]) => void;
  }
}

export function GoogleAnalyticsPageView({ gaId }: { gaId: string }) {
  const pathname = usePathname();
  const searchParams = useSearchParams();

  useEffect(() => {
    if (!pathname || !window.gtag) return;
    const query = searchParams.toString();
    window.gtag("config", gaId, {
      page_path: query ? `${pathname}?${query}` : pathname,
    });
  }, [pathname, searchParams, gaId]);

  return null;
}
