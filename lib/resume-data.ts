export type ResumeType = {
  label: string;
  path: string;
  icon?: string;
};

export const resumes: ResumeType[] = [
  {
    label: "Cyber Security Resume",
    path: "/assets/CyberSecurity.pdf",
  },
  {
    label: "AI & ML Resume",
    path: "/assets/AI_ML.pdf",
  },
  {
    label: "Full Stack Resume",
    path: "/assets/Full_stack.pdf",
  },
  {
    label: "R&D Resume",
    path: "/assets/R&D.pdf",
  }
];
