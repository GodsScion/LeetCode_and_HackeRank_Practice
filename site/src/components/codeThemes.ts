/**
 * The code themes the site ships, and the single source of truth for their keys.
 *
 * Every theme is highlighted into the SAME markup at build time: Shiki's
 * multi-theme mode writes one CSS custom property per theme onto each token
 * span (`--s-vs`, `--s-vt`, …) instead of an inline `color`, so switching is
 * pure CSS and needs no client-side highlighter. See `highlighter.ts`.
 *
 * KEYS ARE DELIBERATELY TWO CHARACTERS. The key is repeated inside the `style`
 * attribute of every single token span on the page — five times over — so each
 * extra character costs real bytes on a long solution page.
 *
 * `surface` is the theme's own editor background. It is duplicated in
 * `global.css` as `--code-bg` under `[data-code-theme='<key>']`, because CSS
 * cannot read it from here and the value on the `<pre>` cannot cascade back up
 * to the card that wraps it. `assertThemeSurfaces()` in `highlighter.ts` fails
 * the build if these drift from what Shiki actually loaded — the CSS side is
 * checked by eye, and the list is five lines long.
 */

export interface CodeTheme {
  /** Short key: the `--s-<key>` variable suffix and the `data-code-theme` value. */
  key: string;
  /** Shiki bundled theme id. */
  id: string;
  /** What the picker shows. */
  label: string;
  /** The theme's own editor background, mirrored into `global.css`. */
  surface: string;
}

/**
 * VS Code's `dark_modern.json` is `{ "include": "./dark_plus.json", "colors": … }`
 * — it carries no `tokenColors` of its own and only restyles workbench chrome.
 * Shiki's bundled `dark-plus` therefore *is* Dark Modern's syntax palette. The
 * one visible difference is the canvas: Dark Modern overrides
 * `editor.background` to #1F1F1F where Dark+ leaves it #1E1E1E, so the surface
 * below is Dark Modern's value, not Shiki's.
 */
export const CODE_THEMES = [
  { key: 'vs', id: 'dark-plus', label: 'VS Code Dark Modern', surface: '#1f1f1f' },
  { key: 'vt', id: 'vitesse-dark', label: 'Vitesse Dark', surface: '#121212' },
  { key: 'gh', id: 'github-dark-default', label: 'GitHub Dark', surface: '#0d1117' },
  { key: 'tn', id: 'tokyo-night', label: 'Tokyo Night', surface: '#1a1b26' },
  { key: 'mk', id: 'monokai', label: 'Monokai', surface: '#272822' },
] as const satisfies readonly CodeTheme[];

export type CodeThemeKey = (typeof CODE_THEMES)[number]['key'];

/** The theme rendered when nothing is stored — and the one plain CSS applies. */
export const DEFAULT_CODE_THEME: CodeThemeKey = 'vs';

/** localStorage key holding the reader's choice. */
export const CODE_THEME_STORAGE_KEY = 'savign:code-theme';

/** Prefix for the per-token CSS variables Shiki emits. Kept short on purpose. */
export const CSS_VAR_PREFIX = '--s-';
