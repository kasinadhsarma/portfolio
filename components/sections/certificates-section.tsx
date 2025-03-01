'use client'

import { certificates, cloudBadges } from "@/data/certificates";
import { CertificatesCarousel } from "@/components/ui/certificates-carousel";

export function CertificatesSection() {
  const featuredCerts = certificates.filter(cert => cert.category === "featured");
  const practicalCerts = certificates.filter(cert => cert.category === "practical");
  
  return (
    <section className="py-16 bg-gradient-to-b from-background to-accent/5">
      <div className="container max-w-6xl space-y-16">
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
          <CertificatesCarousel certificates={cloudBadges} />
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
