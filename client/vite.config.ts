import { fileURLToPath, URL } from 'node:url';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  // Load .env from the repo root so client and server share one file.
  const env = loadEnv(mode, fileURLToPath(new URL('..', import.meta.url)), '');
  const apiTarget = env.VITE_API_BASE_URL || `http://localhost:${env.PORT || 8787}`;

  return {
    plugins: [react()],
    envDir: fileURLToPath(new URL('..', import.meta.url)),
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url)),
      },
    },
    server: {
      port: 5173,
      strictPort: false,
      proxy: {
        // Proxy in dev so the browser sees a same-origin API — no CORS
        // preflight on every SSE request.
        '/api': {
          target: apiTarget,
          changeOrigin: true,
          // Server-sent events must not be buffered by the proxy.
          configure: (proxy) => {
            proxy.on('proxyRes', (proxyRes) => {
              if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
                proxyRes.headers['cache-control'] = 'no-cache, no-transform';
              }
            });
          },
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: true,
      rollupOptions: {
        output: {
          // Split the heavy, rarely-changing libraries so the app chunk stays
          // small and cacheable.
          manualChunks: {
            react: ['react', 'react-dom', 'react-router-dom'],
            charts: ['recharts'],
            motion: ['framer-motion'],
            markdown: ['react-markdown', 'remark-gfm'],
          },
        },
      },
    },
  };
});
