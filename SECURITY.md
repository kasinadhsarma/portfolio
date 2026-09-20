# Security Policy

This is a personal portfolio site (Next.js + Sanity CMS), deployed continuously from the `main` branch. There is no version matrix to track — only the code currently live at [kasinadhsarma.in](https://kasinadhsarma.in) is supported, and fixes land by shipping a new commit to `main` rather than a backported patch release.

## Supported Versions

| Branch | Supported |
| --- | --- |
| `main` (live at kasinadhsarma.in) | :white_check_mark: |
| any fork / older commit | :x: |

## Reporting a Vulnerability

If you find a security issue (XSS, exposed secrets, auth/authorization bypass on the `/studio` route, dependency vulnerability, etc.), please report it privately rather than opening a public issue:

- Email: **kasinadhsarma@gmail.com** — include steps to reproduce, affected URL/route, and impact.
- Alternatively, use [GitHub's private vulnerability reporting](https://github.com/kasinadhsarma/portfolio/security/advisories/new) for this repository if enabled.

**Response expectations:**
- Acknowledgement within a few days.
- If confirmed, a fix is prioritized ahead of feature work since this ships continuously — there's no release train to wait on.
- Please don't test destructive payloads (data deletion, DoS) against the live Sanity dataset or production site; open a local clone against your own Sanity project instead.

## Scope Notes

- The Sanity Studio (`/studio`) is authenticated via Sanity's own auth — report any way to reach write access (`sanity/lib/writeClient.ts`) without valid credentials.
- Client-side secrets: this app should never ship a Sanity **write** token to the browser. Flag it if one is found in any bundled `NEXT_PUBLIC_*` env var or client component.
