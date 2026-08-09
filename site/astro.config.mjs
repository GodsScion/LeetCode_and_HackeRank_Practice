// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

// Published at https://godsscion.github.io/Syntax-Shenanigans
// `base` must match the repo name for GitHub Pages project sites.
export default defineConfig({
  site: 'https://godsscion.github.io',
  base: '/Syntax-Shenanigans',
  trailingSlash: 'always',
  vite: {
    plugins: [tailwindcss()],
  },
});
