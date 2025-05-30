/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    css: true,
  },
  // server: { // Keep server config if it was there for other reasons
  //   proxy: {
  //     '/graphql': 'http://127.0.0.1:5000'
  //   }
  // }
});
