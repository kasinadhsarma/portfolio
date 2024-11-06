/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#6366f1",
        secondary: "#4f46e5",
        accent: "#818cf8",
        background: "#1e1b4b",
        foreground: "#e2e8f0",
        muted: "#334155",
        "muted-foreground": "#94a3b8",
        card: "#1e1b4b",
        "card-foreground": "#e2e8f0",
      },
      animation: {
        blob: "blob 7s infinite",
        glitch: "glitch 5s infinite linear alternate-reverse",
        "glitch-2": "glitch-2 1s infinite linear alternate-reverse",
        blink: "blink 1s step-end infinite",
        "skill-progress": "skill-progress 1s ease-in-out",
      },
      keyframes: {
        blob: {
          "0%": { transform: "translate(0px, 0px) scale(1)" },
          "33%": { transform: "translate(30px, -50px) scale(1.1)" },
          "66%": { transform: "translate(-20px, 20px) scale(0.9)" },
          "100%": { transform: "translate(0px, 0px) scale(1)" },
        },
        glitch: {
          "0%": { clip: "rect(31px, 9999px, 94px, 0)" },
          "4.166666667%": { clip: "rect(91px, 9999px, 43px, 0)" },
          "8.333333333%": { clip: "rect(15px, 9999px, 13px, 0)" },
          "12.5%": { clip: "rect(75px, 9999px, 57px, 0)" },
          "16.66666667%": { clip: "rect(83px, 9999px, 66px, 0)" },
          "20.83333333%": { clip: "rect(63px, 9999px, 24px, 0)" },
          "25%": { clip: "rect(47px, 9999px, 95px, 0)" },
          "100%": { clip: "rect(99px, 9999px, 54px, 0)" },
        },
        "glitch-2": {
          "0%": { clip: "rect(65px, 9999px, 99px, 0)" },
          "50%": { clip: "rect(40px, 9999px, 24px, 0)" },
          "100%": { clip: "rect(87px, 9999px, 59px, 0)" },
        },
        blink: {
          "from, to": { opacity: "1" },
          "50%": { opacity: "0" },
        },
        "skill-progress": {
          "0%": { width: "0%" },
          "100%": { width: "var(--progress)" },
        },
      },
    },
  },
  plugins: [
    require("@tailwindcss/forms"),
    function({ addUtilities }) {
      const newUtilities = {
        '.text-shadow-glitch': {
          textShadow: '-2px 0 #ff00c1',
        },
        '.text-shadow-glitch2': {
          textShadow: '-2px 0 #00fff9, 2px 2px #ff00c1',
        },
      }
      addUtilities(newUtilities)
    }
  ],
}
