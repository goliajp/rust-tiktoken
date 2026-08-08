// Separate config so the competitive benchmark never lands in the site build:
// it bundles two rival tokenizers and their vocabularies (~6.5 MB) purely to
// be measured. `npm run bench` builds and drives it; `npm run build` cannot
// see it.
import { defineConfig } from 'vite'

export default defineConfig({
  root: 'bench',
  build: { outDir: '../.bench-dist', emptyOutDir: true, chunkSizeWarningLimit: 16000 },
})
