/**
 * vite.config.ts --- Vite configuration for the benchmark dashboard
 *
 * Contains:
 *   default export: dev/build settings with the React plugin
 */

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^[/]api/, ""),
      },
    },
  },
});
