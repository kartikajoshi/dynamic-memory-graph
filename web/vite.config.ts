import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev: Vite on 5173 proxies /api to the FastAPI server (port 8000 by default; DMG_API_PORT overrides).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': `http://127.0.0.1:${process.env.DMG_API_PORT || 8000}` } },
  build: { target: 'es2022', chunkSizeWarningLimit: 900 },
});
