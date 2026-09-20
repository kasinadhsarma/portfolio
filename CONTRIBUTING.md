# Contributing

This is Kasinadh Sarma's personal portfolio site — not a general-purpose open-source project looking for feature contributions. That said, bug reports and small fixes (typos, broken links, accessibility issues, build errors) are welcome.

## Before You Open a PR

For anything beyond a trivial fix, open an issue first describing what you found and how you'd fix it — this avoids spending effort on a PR that doesn't fit the site's direction (content and design are personal/opinionated by nature here).

## Local Setup

```bash
npm install
npm run dev        # starts Next.js dev server
```

You'll need your own Sanity project for content to load — see [docs/db/SANITY_SCHEMA.md](./docs/db/SANITY_SCHEMA.md) for the schema, and set `NEXT_PUBLIC_SANITY_PROJECT_ID` / `NEXT_PUBLIC_SANITY_DATASET` in `.env.local` (see `sanity/env.ts`).

Other useful commands:

```bash
npm run lint        # ESLint
npm run typecheck   # tsc --noEmit
npm run build       # production build (includes obfuscation step, see scripts/obfuscate-build.mts)
```

## Project Structure

Route files under `app/` are thin re-exports; real page markup lives in `components/pages/<page>/`, shared UI primitives in `components/ui/`, and all Tailwind class strings live in `lib/responsive/pattrens/*.ts` (never inline in a component — see [docs/maths/UI_COMPONENTS.md](./docs/maths/UI_COMPONENTS.md) and [docs/maths/LAYOUT_DIMENSIONS.md](./docs/maths/LAYOUT_DIMENSIONS.md)). See [docs/architecture/README.md](./docs/architecture/README.md) for the full diagram.

## Reporting Security Issues

Do not open a public issue for a security vulnerability — see [SECURITY.md](./SECURITY.md).

## Code Style

- Follow the existing pattern-file convention for styling (no inline Tailwind strings in components).
- Keep PRs focused — one fix or one small feature per PR.
- Run `npm run lint` and `npm run typecheck` before opening a PR.
