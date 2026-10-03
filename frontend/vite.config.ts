import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  // Public production builds are browser-only; opt into Jev with --mode dual.
  define: { __KEEPUP_BROWSER_ONLY__: JSON.stringify(mode !== 'development' && mode !== 'dual') },
  build: { outDir: mode === 'dual' ? 'dist-local' : 'dist' },
}));
