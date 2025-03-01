export type ResumeType = {
  label: string;
  path: string;
  icon?: string;
};

export const resumes: ResumeType[] = [
  {
    label: "Cyber Security Resume",
    path: "/pdf/CyberSecurity.pdf",
  },
  {
    label: "AI & ML Resume",
    path: "/pdf/AI_ML.pdf",
  },
  {
    label: "Full Stack Resume",
    path: "/pdf/Full_stack.pdf",
  },
  {
    label: "R&D Resume",
    path: "/pdf/R&D.pdf",
  }
];
