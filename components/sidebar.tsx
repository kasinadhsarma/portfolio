"use client";

import Image from "next/image";
import Link from "next/link";
import { Mail, Phone, Calendar, MapPin } from "lucide-react";
import { cn } from "@/lib/utils";
import { useState } from "react";

const socialLinks = [
  { href: "https://www.facebook.com/s.kasinadh.1/", icon: "facebook" },
  { href: "https://twitter.com/kasinadhsarma", icon: "twitter" },
  { href: "https://www.instagram.com/skasinadh/", icon: "instagram" },
  { href: "https://www.linkedin.com/in/kasinadhsarma/", icon: "linkedin" },
  { href: "https://github.com/kasinadhsarma", icon: "github" },
  { href: "https://linktr.ee/kasinadhsarma", icon: "link" }
];

export function Sidebar() {
  const [showContacts, setShowContacts] = useState(false);

  return (
    <aside className="w-72 flex-shrink-0 border-r border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex flex-col h-full px-6 py-8">
        <div className="flex flex-col items-center text-center mb-8">
          <div className="relative w-20 h-20 mb-4">
            <Image
              src="/img/me.jpg"
              alt="Kasinadh Sarma"
              fill
              className="rounded-full object-cover"
              priority
            />
          </div>
          <h1 className="text-xl font-semibold mb-1">Kasinadh Sarma</h1>
          <p className="text-sm text-muted-foreground">Cyber Security Researcher</p>
          
          <button
            onClick={() => setShowContacts(!showContacts)}
            className="mt-4 flex items-center gap-2 px-4 py-2 rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
          >
            <span>{showContacts ? "Hide Contacts" : "Show Contacts"}</span>
            <svg
              className={cn(
                "h-4 w-4 transition-transform",
                showContacts ? "rotate-180" : ""
              )}
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>
        </div>

        {showContacts && (
          <div className="space-y-6 mb-8">
            <div className="h-px bg-border" />
            <ul className="space-y-4">
              <li className="flex items-center gap-3">
                <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10">
                  <Mail className="h-4 w-4 text-primary" />
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-muted-foreground">Email</span>
                  <a
                    href="mailto:kasinadhsarma@gmail.com"
                    className="text-sm hover:text-primary transition-colors"
                  >
                    kasinadhsarma@gmail.com
                  </a>
                </div>
              </li>
              <li className="flex items-center gap-3">
                <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10">
                  <Phone className="h-4 w-4 text-primary" />
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-muted-foreground">Phone</span>
                  <a
                    href="tel:+916305953487"
                    className="text-sm hover:text-primary transition-colors"
                  >
                    +91 6305953487
                  </a>
                </div>
              </li>
              <li className="flex items-center gap-3">
                <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10">
                  <Calendar className="h-4 w-4 text-primary" />
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-muted-foreground">Birthday</span>
                  <time dateTime="2002-12-22" className="text-sm">
                    December 22, 2002
                  </time>
                </div>
              </li>
              <li className="flex items-center gap-3">
                <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10">
                  <MapPin className="h-4 w-4 text-primary" />
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-muted-foreground">Location</span>
                  <address className="text-sm not-italic">
                    Guntur, Andhra Pradesh, India
                  </address>
                </div>
              </li>
            </ul>
            <div className="h-px bg-border" />
            <div className="flex justify-center gap-3">
              {socialLinks.map((link) => (
                <Link
                  key={link.icon}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10 hover:bg-primary/20 text-primary transition-colors"
                >
                  <i className={`bx bxl-${link.icon}`}></i>
                </Link>
              ))}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
