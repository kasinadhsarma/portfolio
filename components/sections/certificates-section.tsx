'use client'

import { useEffect, useState } from "react";
import { SanityCertificateCard } from "@/types/sanity";
import { CertificatesCarousel } from "@/components/ui/certificates-carousel";

export function CertificatesSection() {
  const [certificates, setCertificates] = useState<SanityCertificateCard[]>([]);

  useEffect(() => {
    fetch("/api/certificates")
      .then((res) => res.json())
      .then((data) => setCertificates(Array.isArray(data) ? data : []))
      .catch((error) => console.error("Failed to load certificates:", error));
  }, []);

  const featuredCerts = certificates.filter(cert => cert.category === "featured");
  const cloudCerts = certificates.filter(cert => cert.category === "cloud");
  const practicalCerts = certificates.filter(cert => cert.category === "practical");

  return (
    <section className="py-16 bg-gradient-to-b from-background to-accent/5">
      <div className="container max-w-6xl 2xl:max-w-[1600px] space-y-16">
        {/* Featured Certifications */}
        <div>
          <h2 className="text-3xl font-bold mb-8 bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent">
            Featured Certifications
          </h2>
          <CertificatesCarousel certificates={featuredCerts} />
        </div>

        {/* Google Cloud Badges */}
        <div>
          <h2 className="text-3xl font-bold mb-8 bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent">
            Google Cloud Badges
          </h2>
          <CertificatesCarousel certificates={cloudCerts} />
        </div>

        {/* Practical Experience */}
        {practicalCerts.length > 0 && (
          <div>
            <h2 className="text-3xl font-bold mb-8 bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent">
              Practical Experience
            </h2>
            <CertificatesCarousel certificates={practicalCerts} />
          </div>
        )}
      </div>
    </section>
  );
}
