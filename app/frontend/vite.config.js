import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Relative "/api" calls are proxied to FastAPI (:8000) in dev and are same-origin
// in prod (FastAPI serves the built SPA). No API base URL hardcoded, no CORS to manage.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
