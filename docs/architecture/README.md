# Architecture Diagrams

## Portfolio App Architecture

Actual structure of this repo (Next.js App Router + Sanity CMS):

```mermaid
graph TD
    subgraph Client["Browser"]
        Pages["app/*/page.tsx routes"]
    end

    subgraph Next["Next.js App"]
        Pages --> PageComponents["components/pages/*"]
        PageComponents --> UI["components/ui/*"]
        PageComponents --> Widgets["components/widgets/*"]
        PageComponents --> Layout["components/layout/*"]
        UI --> Patterns["lib/responsive/pattrens/*.ts\n(Tailwind class patterns)"]
        PageComponents --> Patterns
        UI --> Utils["lib/utils.ts (cn, tailwind-merge)"]
        Providers["components/providers/*"] --> Pages
    end

    subgraph Data["Content Layer"]
        SanityClient["sanity/lib/client.ts (GROQ, read)"]
        WriteClient["sanity/lib/writeClient.ts (write)"]
        Schema["sanity/schemaTypes/* (project, certificate,\nachievement, skillCategory, resumeFile, resume)"]
    end

    PageComponents --> SanityClient
    SanityClient --> Schema
    Studio["app/studio/[[...tool]]/page.tsx\n(Sanity Studio UI)"] --> Schema
    Studio --> WriteClient

    subgraph Config["Styling / Build Config"]
        Tailwind["tailwind.config.ts"]
        Globals["app/globals.css (CSS vars, keyframes)"]
    end

    Patterns --> Tailwind
    Patterns --> Globals
```

See [LAYOUT_DIMENSIONS.md](../maths/LAYOUT_DIMENSIONS.md) and [UI_COMPONENTS.md](../maths/UI_COMPONENTS.md) for the per-page and per-component breakdown of this diagram.

## Sanity Schema (Data Layer)

The `Schema` node above expands into these document types — see [db/SANITY_SCHEMA.md](../db/SANITY_SCHEMA.md) for the full field-level reference.

```mermaid
erDiagram
    PROJECT {
        string title
        slug slug
        text description
        array category
        string status
        url github
        url liveUrl
        datetime publishedAt
    }
    PROJECT ||--o{ TEAM_MEMBER : "team[]"
    TEAM_MEMBER {
        string name
        string role
        url url
    }

    CERTIFICATE {
        string title
        string issuer
        string date
        string category
        number order
    }

    ACHIEVEMENT {
        string title
        text description
        string date
        number order
    }

    SKILL_CATEGORY {
        string category
        array items
        number order
    }

    RESUME_FILE {
        string label
        file file
        number order
    }

    RESUME {
        string id "singleton, fixed id = resume"
    }
    RESUME ||--o{ EDUCATION_ENTRY : "education[]"
    RESUME ||--o{ EXPERIENCE_ENTRY : "experience[]"
    RESUME ||--o{ RESUME_PROJECT_ENTRY : "projects[]"
    RESUME ||--o{ TRAINING_ENTRY : "training[]"
    EDUCATION_ENTRY {
        string institution
        string period
        text description
    }
    EXPERIENCE_ENTRY {
        string title
        string organization
        string period
        boolean current
        array highlights
    }
    RESUME_PROJECT_ENTRY {
        string title
        string subtitle
        array highlights
    }
    TRAINING_ENTRY {
        string title
        string organization
        string period
        text description
    }
```

`CERTIFICATE`, `ACHIEVEMENT`, and `SKILL_CATEGORY` have no relations to other document types — they're standalone collections. `TEAM_MEMBER` and the `*_ENTRY` types are embedded objects (arrays-of-objects), not Sanity `reference` fields, so they only ever exist nested inside their parent document.

---

## Legacy Diagrams (unrelated to this codebase)

> **Note:** the two diagrams below (`mermaid-diagram-2025-02-27-064656.png`, `mermaid-diagram-2025-02-27-064818.svg`) depict a fictional/example hardware architecture — photonic interconnects, SPU clusters, quantum-annealing schedulers, neuromorphic grids. They do **not** describe this Next.js portfolio's design. They're kept here as historical assets rather than deleted; treat them as reference material only, not as documentation of this app.

![Legacy architecture diagram 1](./mermaid-diagram-2025-02-27-064656.png)

[mermaid-diagram-2025-02-27-064818.svg](./mermaid-diagram-2025-02-27-064818.svg) (open directly — large SVG, not inlined here).
