import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Plotly is ~1 MB of the bundle and changes only when the dependency
        // does, so it is split out to cache independently of app code. It
        // cannot be shrunk further without dropping the charts.
        manualChunks: { plotly: ["plotly.js-basic-dist-min"] },
      },
    },
    chunkSizeWarningLimit: 1200,
  },
  server: {
    port: 5173,
    // The API allowlists this origin for CORS, so the dev server proxies rather
    // than the browser making cross-origin calls with credentials.
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
