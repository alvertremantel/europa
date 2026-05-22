import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Required because circuitsvis@1.43.3 package.json "exports" map
      // points to non-existent dist/index.esm.mjs / dist/index.cjs.js;
      // the actual ES module entry is dist/module/index.js.
      'circuitsvis': path.resolve(__dirname, 'node_modules/circuitsvis/dist/module/index.js')
    }
  },
  build: {
    rolldownOptions: {
      output: {
        // Split the lazy-loaded circuitsvis chunk so that TensorFlow.js
        // (@tensorflow/tfjs) ships in its own deferred chunk instead of
        // bloating the attention-visualization chunk above 500 kB.
        codeSplitting: {
          groups: [
            {
              name: 'tensorflow',
              test: /node_modules\/@tensorflow/,
            },
            {
              name: 'circuitsvis',
              test: /node_modules\/circuitsvis/,
            },
          ],
        },
      },
    },
    // Keep this limit for the deferred TensorFlow chunk only; it is not a
    // general suppression of entry-chunk growth. @tensorflow/tfjs is large
    // and only loads when the activation panel renders, so the initial
    // dashboard shell stays small.
    chunkSizeWarningLimit: 900,
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
