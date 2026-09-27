import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig(({ isSsrBuild }) => ({
  plugins: [svelte()],
  base: '/airhockey-distillation/',
  build: { copyPublicDir: !isSsrBuild },
}));
