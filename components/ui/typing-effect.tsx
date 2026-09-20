"use client";

import * as React from "react";
import { typingEffectPatterns as p } from "@/lib/responsive/pattrens/ui";

export function TypingEffect({ texts }: { texts: string[] }) {
  const [currentTextIndex, setCurrentTextIndex] = React.useState(0);
  const [currentText, setCurrentText] = React.useState("");
  const [isDeleting, setIsDeleting] = React.useState(false);

  React.useEffect(() => {
    const timeout = setTimeout(() => {
      const fullText = texts[currentTextIndex];
      if (!isDeleting) {
        setCurrentText(fullText.substring(0, currentText.length + 1));
        if (currentText === fullText) {
          setIsDeleting(true);
          setTimeout(() => {}, 1000);
        }
      } else {
        setCurrentText(fullText.substring(0, currentText.length - 1));
        if (currentText === "") {
          setIsDeleting(false);
          setCurrentTextIndex((currentTextIndex + 1) % texts.length);
        }
      }
    }, isDeleting ? 50 : 100);
    return () => clearTimeout(timeout);
  }, [currentText, currentTextIndex, isDeleting, texts]);

  return (
    <span className={p.text}>
      {currentText}
      <span className={p.cursor}>|</span>
    </span>
  );
}
