'use client'

import { useEffect, useState } from "react";
import { SanityCertificateCard } from "@/types/sanity";
import { CertificatesCarousel } from "@/components/ui/certificates-carousel";
import { certificatesPatterns as p } from "@/lib/responsive/pattrens/home";

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
    <section className={p.section}>
      <div className={p.container}>
        {/* Featured Certifications */}
        <div>
          <h2 className={p.heading}>
            Featured Certifications
          </h2>
          <CertificatesCarousel certificates={featuredCerts} priorityFirst />
        </div>

        {/* Google Cloud Badges */}
        <div>
          <h2 className={p.heading}>
            Google Cloud Badges
          </h2>
          <CertificatesCarousel certificates={cloudCerts} />
        </div>

        {/* Practical Experience */}
        {practicalCerts.length > 0 && (
          <div>
            <h2 className={p.heading}>
              Practical Experience
            </h2>
            <CertificatesCarousel certificates={practicalCerts} />
          </div>
        )}
      </div>
    </section>
  );
}
