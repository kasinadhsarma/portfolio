import { projectsHeaderPatterns as p } from "@/lib/responsive/pattrens/projects"

export function ProjectsHeader() {
  return (
    <div className={p.wrapper}>
      <div>
        <div className={p.inner}>
          <div className={p.titleRow}>
            <h1 className={p.title}>Projects</h1>
          </div>
          <div className={p.metaRow}>
            <p className={p.description}>
              Explore my portfolio of personal and professional projects
            </p>
            <a
              href="https://wakatime.com/@9849b760-c9b2-46e7-b469-271f5faa6c63"
              target="_blank"
              rel="noopener noreferrer"
              className={p.badgeLink}
            >
              <img
                src="https://wakatime.com/badge/user/9849b760-c9b2-46e7-b469-271f5faa6c63.svg"
                alt="Total time coded since Aug 17 2024"
                height="20"
              />
            </a>
          </div>
          <div className={p.divider}></div>
        </div>
      </div>
    </div>
  );
}
