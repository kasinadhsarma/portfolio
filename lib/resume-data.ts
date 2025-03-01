export type ResumeType = {
  label: string;
  path: string;
  icon?: string;
};

export const resumes: ResumeType[] = [
  {
    label: "Cyber Security Resume",
    path: "public/pdf/CyberSecurity.pdf",
  },
  {
    label: "AI & ML Resume",
    path: "public/pdf/AI_ML.pdf",
  },
  {
    label: "Full Stack Resume",
    path: "/public/pdf/Full_stack.pdf",
  },
  {
    label: "R&D Resume",
    path: "/public/pdf/R&D.pdf",
  }
];
