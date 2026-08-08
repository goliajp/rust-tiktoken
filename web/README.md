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

Any static host works — `dist/` has no server-side requirements. Two things
matter for the wasm payload:

1. **MIME type**: `.wasm` must be served as `application/wasm`
   (Cloudflare Pages / Netlify / Vercel all do this out of the box; for nginx
   add `types { application/wasm wasm; }`).
2. **Cache**: everything under `dist/assets/` is content-hashed — serve it with
   `Cache-Control: public, max-age=31536000, immutable` so the 11 MB wasm is
   fetched once per version. `index.html` should stay `no-cache`.

Cloudflare Pages example:

```bash
cd web && npm install && npm run build
# then point a Pages project at web/ with:
#   build command: npm run build
#   output directory: dist
# and add tiktoken.golia.jp as the custom domain.
```

nginx example:

```nginx
server {
  server_name tiktoken.golia.jp;
  root /srv/tiktoken-web/dist;
  types { application/wasm wasm; }
  location /assets/ {
    add_header Cache-Control "public, max-age=31536000, immutable";
  }
  location / {
    add_header Cache-Control "no-cache";
    try_files $uri /index.html;
  }
}
```

## Content maintenance

The tables in `src/components/Tables.tsx` (encoding spec sheet, benchmark
numbers) and the stat figures in `App.tsx` / `i18n.ts` mirror the repository
README — they are the site's only hand-maintained data. When a release adds
encodings or changes headline numbers, update them together with the READMEs.
The encoding picker itself needs no maintenance: it enumerates whatever the
installed wasm package reports via `listEncodings()`.
