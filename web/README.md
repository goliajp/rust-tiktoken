# web — tiktoken.golia.jp

The project site: an introduction to `tiktoken` plus a live, fully client-side
tokenizer playground powered by [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) —
the same wasm package this repository publishes, running unmodified. Trilingual
(English / 简体中文 / 日本語) with automatic language detection and a manual
switch persisted to `localStorage`.

## Stack

- **Vite + React + TypeScript** — no CSS framework; the design system is
  ~500 lines of hand-written CSS in `src/styles.css` (instrument-panel
  aesthetic: dark ground, phosphor accent, Space Mono / Instrument Sans).
- **`@goliapkg/tiktoken-wasm`** — loaded lazily on first paint; all 17
  encodings run in-browser, nothing the visitor types is uploaded.
- **i18n** — a flat trilingual dictionary in `src/i18n.ts` with a React
  context. No library: three locales, one page.

```
src/
├── main.tsx               entry
├── App.tsx                page shell: hero, sections, footer, language switch
├── i18n.ts                en/zh/ja dictionary + detection
├── tokenizer.ts           wasm init + encoding cache + token→segment resolution
├── styles.css             the whole design system
└── components/
    ├── Playground.tsx     encoding picker, input, meters, colored token view
    ├── Tables.tsx         encoding spec sheet + benchmark table
    └── CodeBlock.tsx      copyable install/usage snippets
```

## Develop

```bash
cd web
npm install
npm run dev        # http://localhost:5173
```

## Build

```bash
npm run build      # type-checks, then emits dist/
npm run preview    # serve dist/ locally
```

The build output is fully static. The wasm binary is ~11 MB (17 zstd-compressed
vocabularies are embedded; zstd data does not gzip further) — the UI shows a
loading state and the page itself paints instantly, so this is a
first-interaction cost, not a first-paint cost.

## Deploy (tiktoken.golia.jp)

The site is live at **<https://tiktoken.golia.jp>**, served by Caddy on `t01`
from `/apps/tiktoken/web`.

```bash
web/deploy.sh            # build → rsync to t01 → verify the live origin
web/deploy.sh --check    # build and print the wasm digest; upload nothing
```

The script verifies the deploy rather than assuming it: the origin must return
200, the wasm must come back as `application/wasm`, and the served wasm's
SHA-256 must equal the one just built. A truncated upload still answers 200, so
the digest check is the one that actually matters.

### What devops owns (do not re-create here)

The box, TLS, the vhost and DNS belong to [`goliajp/devops`](https://github.com/goliajp/devops),
where they are database rows reconciled onto the device — **not** files to edit
by hand. They are already provisioned:

| Resource | Where it lives | Value |
|---|---|---|
| DNS | `dns` store, zone `golia.jp` | `tiktoken` CNAME → `t01.golia.jp.` |
| vhost | `caddy_sites` store, id `tiktoken` | domain `tiktoken.golia.jp`, root `/apps/tiktoken/web` |
| TLS | Caddy ACME | automatic |

`/etc/caddy/Caddyfile` on `t01` is generated from that store and carries a
"do not edit manually" header; editing it in place is reverted by the next
reconcile. To change the vhost, update the store and redeploy:

```bash
devops caddy list                     # confirm the tiktoken row
devops caddy drift t01                # must be clean before deploying
devops caddy deploy t01               # regenerate + push + validate + reload
```

Note the `block` column holds the **body only** — the generator emits the
`domain { … }` wrapper and the indentation itself. Including the domain line in
the body produces `unrecognized directive: tiktoken.golia.jp` at Caddy's config
validation step (which refuses the deploy and leaves the live config untouched).

The vhost sets the two headers this site needs: hashed assets under `/assets/*`
get `max-age=31536000, immutable` (the 11 MB wasm is fetched once per release),
everything else gets `no-cache`. Caddy serves `.wasm` as `application/wasm`
natively — no MIME configuration is required.

### Hosting elsewhere

`dist/` is plain static output with no server-side requirements. On any other
host, reproduce just those two things: `application/wasm` for `.wasm`, and
immutable caching for the content-hashed `/assets/*` while `index.html` stays
uncached.

## Content maintenance

The tables in `src/components/Tables.tsx` (encoding spec sheet, benchmark
numbers) and the stat figures in `App.tsx` / `i18n.ts` mirror the repository
README — they are the site's only hand-maintained data. When a release adds
encodings or changes headline numbers, update them together with the READMEs.
The encoding picker itself needs no maintenance: it enumerates whatever the
installed wasm package reports via `listEncodings()`.
