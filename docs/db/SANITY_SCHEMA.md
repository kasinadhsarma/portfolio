# Sanity CMS — Data Schema Reference

This is the "database schema" for the portfolio: Sanity has no relational tables, but each `defineType` in [sanity/schemaTypes/](../../sanity/schemaTypes/) is a document type, and its `defineField` entries are the columns. Schema registry: [sanity/schemaTypes/index.ts](../../sanity/schemaTypes/index.ts). Studio grouping (desk structure): [sanity/structure.ts](../../sanity/structure.ts) — `resume` is pinned as a singleton at the top; every other type gets an auto-generated list.

Query client: [sanity/lib/client.ts](../../sanity/lib/client.ts) · Write client: [sanity/lib/writeClient.ts](../../sanity/lib/writeClient.ts) · TypeScript projections: [types/sanity.ts](../../types/sanity.ts).

## Entity-Relationship Overview

```
project (document)
  └─ team[] (embedded object, no reference)

certificate (document)          — standalone

achievement (document)          — standalone

skillCategory (document)        — standalone

resumeFile (document)           — standalone (one row per downloadable PDF)

resume (document, SINGLETON, id="resume")
  ├─ education[]   (embedded object: educationEntry)
  ├─ experience[]  (embedded object: experienceEntry)
  ├─ projects[]    (embedded object: resumeProjectEntry)
  └─ training[]    (embedded object: trainingEntry)
```

There are no `reference` fields between document types in this schema — all one-to-many relationships (e.g. resume → experience entries) are modeled as **embedded arrays of objects**, not joins.

---

## `project`

File: [sanity/schemaTypes/project.ts](../../sanity/schemaTypes/project.ts) · Icon: `DocumentIcon`

| Field | Type | Notes |
|---|---|---|
| `title` | `string` | required |
| `slug` | `slug` | required; source: `title`, max 96 chars |
| `description` | `text` (4 rows) | short summary |
| `longDescription` | `array` of `block` | Portable Text rich content |
| `image` | `image` (hotspot) | + `alt` string subfield |
| `gallery` | `array` of `image` (hotspot) | + `alt` per image |
| `technologies` | `array` of `string` | tag layout |
| `category` | `array` of `string` | tag layout; enum: `ai`, `web`, `cybersecurity`, `database`, `cloud`, `mobile`, `desktop`, `other`; required, 1–3 selections |
| `status` | `string` | enum: `development` (default), `completed`, `on-hold`, `archived` |
| `github` | `url` | http/https only |
| `liveUrl` | `url` | http/https only |
| `startDate` | `date` | |
| `endDate` | `date` | |
| `team` | `array` of `object` | inline fields: `name` (string), `role` (string), `url` (url) |
| `publishedAt` | `datetime` | defaults to `now()` |

**Orderings:** Published Date (new/old), Title A–Z. **Preview:** title / image / category-or-description.

**Corresponding TS type:** `SanityProjectCard` in `types/sanity.ts` (a flattened read-projection, not a 1:1 mirror — e.g. `image` becomes a resolved URL string).

---

## `certificate`

File: [sanity/schemaTypes/certificate.ts](../../sanity/schemaTypes/certificate.ts) · Icon: `StarFilledIcon`

| Field | Type | Notes |
|---|---|---|
| `title` | `string` | required |
| `issuer` | `string` | required |
| `date` | `string` | free text — a year *or* a status like `"Active"` |
| `image` | `image` (hotspot) | required; + `alt` string subfield |
| `url` | `url` | http/https only; credential verification link |
| `category` | `string` | required; enum: `featured`, `cloud`, `work`, `practical` |
| `order` | `number` | default `0`; lower = earlier in list |

**Orderings:** Display Order asc. **Preview:** title / issuer / image.
**TS types:** `SanityCertificate` (raw), `SanityCertificateCard` (read-projection).

---

## `achievement`

File: [sanity/schemaTypes/achievement.ts](../../sanity/schemaTypes/achievement.ts) · Icon: `StarIcon`

| Field | Type | Notes |
|---|---|---|
| `title` | `string` | required |
| `icon` | `image` (hotspot) | |
| `description` | `text` (3 rows) | |
| `date` | `string` | free text |
| `order` | `number` | default `0` |

**Orderings:** Display Order asc. **Preview:** title / date / icon.

---

## `skillCategory`

File: [sanity/schemaTypes/skillCategory.ts](../../sanity/schemaTypes/skillCategory.ts) · Icon: `TagsIcon`

| Field | Type | Notes |
|---|---|---|
| `category` | `string` | required — the group label, e.g. "Languages" |
| `items` | `array` of `string` | tag layout — the individual skills |
| `order` | `number` | default `0` |

**Orderings:** Display Order asc. **Preview:** category / items joined as subtitle.

---

## `resumeFile`

File: [sanity/schemaTypes/resumeFile.ts](../../sanity/schemaTypes/resumeFile.ts) · Icon: `DocumentPdfIcon`

| Field | Type | Notes |
|---|---|---|
| `label` | `string` | required — button text, e.g. "Resume (2025)" |
| `file` | `file` | required; `accept: application/pdf` |
| `order` | `number` | default `0` |

**Orderings:** Display Order asc. **Preview:** label.
**TS type:** `SanityResumeFileCard` (`{ label, url }` — resolved asset URL).

---

## `resume` (singleton)

File: [sanity/schemaTypes/resume.ts](../../sanity/schemaTypes/resume.ts) · Icon: `CaseIcon` · Fixed document id: `resume`

Pinned as a single editable document in the Studio (see `sanity/structure.ts`) — there is only ever one `resume` document, unlike every other type above which is a repeatable collection.

| Field | Type | Embedded object fields |
|---|---|---|
| `education` | `array` of `educationEntry` | `institution` (string), `period` (string), `description` (text, 3 rows) |
| `experience` | `array` of `experienceEntry` | `title` (string), `organization` (string), `period` (string), `current` (boolean, default `false`), `highlights` (array of string) |
| `projects` | `array` of `resumeProjectEntry` | `title` (string), `subtitle` (string), `highlights` (array of string) |
| `training` | `array` of `trainingEntry` | `title` (string), `organization` (string), `period` (string), `description` (text, 3 rows) |

**Preview:** static title `"Resume"` (no `select`, since there's only one document).
**TS types:** `SanityResume` + one interface per embedded object (`SanityEducationEntry`, `SanityExperienceEntry`, `SanityResumeProjectEntry`, `SanityTrainingEntry`).

---

## Field-Type Legend

| Sanity type | Meaning |
|---|---|
| `string` | short text |
| `text` | multi-line text (`rows` controls textarea height in Studio) |
| `slug` | URL-safe string, can auto-derive from another field via `options.source` |
| `number` | numeric |
| `boolean` | checkbox |
| `date` / `datetime` | date picker / date+time picker |
| `url` | string validated as a URL, optionally restricted by `Rule.uri({ scheme })` |
| `image` | asset reference + optional `hotspot` cropping + custom subfields (e.g. `alt`) |
| `file` | arbitrary binary asset reference (used here for the resume PDF) |
| `array` | ordered list; `of` declares the allowed member type(s) |
| `block` | Portable Text rich-content block (used for `project.longDescription`) |
| `object` | inline, unnamed/embedded structure with its own `fields` |

## Conventions Used Across Types

- **Ordering:** Any list-like type (`certificate`, `achievement`, `skillCategory`, `resumeFile`) carries a manual `order: number` field (default `0`) plus an `orderAsc` ordering, so editors control display order in Studio rather than relying on creation time.
- **Alt text:** Every `image` field that appears in public-facing cards defines an `alt` subfield for accessibility.
- **Enums as `options.list`:** Fixed vocabularies (`project.category`, `project.status`, `certificate.category`) are defined inline as `{ title, value }` lists rather than as separate documents/references.
- **No cross-document references:** relationships are flattened into embedded objects/arrays (e.g. `project.team`, `resume.experience`) rather than Sanity `reference` fields — simpler for a single-owner portfolio with no shared/reused sub-entities.

## Editorial Workflow

1. **Open Studio** at the `/studio` route of the deployed site (mounted by `app/studio/[[...tool]]/page.tsx`, configured in `sanity.config.ts`).
2. **Create a document** of the desired type from the sidebar list (all types except `resume`, which is a pinned singleton) and fill in at least the required fields (see the `required` notes per type above).
3. **Publish** — content only appears on the live site once published, not on every save.

Tips:
- Keep `technologies`/tag-style arrays consistently cased (e.g. `"React"`, not `"react"`) since they're rendered as-is.
- Use `order` fields to control display order deliberately — new documents default to `0` and will sort first until adjusted.
- `slug` on `project` auto-generates from `title` but can be edited for cleaner URLs.
