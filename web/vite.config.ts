import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  build: {
    // the wasm vocabularies are large by nature; silence the default warning
    chunkSizeWarningLimit: 16000,
  },
})
