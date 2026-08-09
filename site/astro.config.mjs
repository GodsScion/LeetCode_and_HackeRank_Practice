// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

// Published at https://godsscion.github.io/LeetCode_and_HackeRank_Practice
// `base` must match the repo name for GitHub Pages project sites.
export default defineConfig({
  site: 'https://godsscion.github.io',
  base: '/LeetCode_and_HackeRank_Practice',
  trailingSlash: 'always',
  vite: {
    plugins: [tailwindcss()],
  },
});
