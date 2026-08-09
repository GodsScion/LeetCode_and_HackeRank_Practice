/**
 * ONE Shiki highlighter for the whole build.
 *
 * `createHighlighter()` loads WASM + grammars + themes, which costs ~1s. This
 * module is evaluated once per Vite module graph, so every page reuses the same
 * instance and the same in-flight promise. Never call `codeToHtml` from `shiki`
 * directly — that spins up a fresh highlighter per call.
 *
 * Themes: all five in `codeThemes.ts`, rendered simultaneously. With
 * `defaultColor: false` Shiki drops the inline `color` and writes one CSS
 * custom property per theme onto each token span, so the reader switches theme
 * with a single attribute on `<html>` and no client-side highlighting. The
 * default (VS Code Dark Modern) is the one plain CSS reads; the other four are
 * `[data-code-theme='…']` overrides in `global.css`.
 */
import { createHighlighter, type Highlighter, type ShikiTransformer } from 'shiki';
import { CODE_THEMES, CSS_VAR_PREFIX } from './codeThemes.ts';

/** `{ vs: 'dark-plus', vt: 'vitesse-dark', … }`, the shape `codeToHtml` wants. */
const THEME_MAP = Object.fromEntries(CODE_THEMES.map((t) => [t.key, t.id])) as Record<
  string,
  string
>;

/** Only the grammars this repo actually contains. */
const LANGS = ['python', 'java', 'javascript', 'typescript', 'c', 'html'] as const;

const SUPPORTED = new Set<string>(LANGS);

/**
 * When one theme italicises a token and another does not, Shiki spells the
 * difference out for *every* theme — `--s-vs-font-style:inherit` and friends —
 * which is ~25 dead bytes per theme per span. The CSS reads these through
 * `var(…, normal)`, so an absent variable already means "inherit". Dropping
 * them is worth low tens of KB on a long page.
 *
 * Only `:inherit` declarations are removed; a colour never ends in `:inherit`.
 */
const dropInheritVars: ShikiTransformer = {
  name: 'savign:drop-inherit-vars',
  span(node) {
    const style = node.properties?.style;
    if (typeof style !== 'string') return;
    const kept = style.split(';').filter((decl) => decl && !decl.endsWith(':inherit'));
    if (kept.length) node.properties.style = kept.join(';');
    else delete node.properties.style;
  },
};

let instance: Promise<Highlighter> | undefined;

function getHighlighter(): Promise<Highlighter> {
  if (!instance) {
    instance = createHighlighter({
      themes: CODE_THEMES.map((t) => t.id),
      langs: [...LANGS],
    }).then((highlighter) => {
      assertThemeSurfaces(highlighter);
      return highlighter;
    });
  }
  return instance;
}

/**
 * `global.css` hard-codes each theme's background because a CSS variable set on
 * the `<pre>` cannot cascade up to the card wrapping it. Fail the build loudly
 * if a Shiki upgrade moves one of those canvases out from under us.
 *
 * Dark Modern is the deliberate exception: it overrides Dark+'s #1E1E1E canvas
 * to #1F1F1F, so we follow VS Code rather than Shiki there.
 */
const SURFACE_EXCEPTIONS = new Set(['dark-plus']);

function assertThemeSurfaces(highlighter: Highlighter): void {
  for (const theme of CODE_THEMES) {
    if (SURFACE_EXCEPTIONS.has(theme.id)) continue;
    const actual = highlighter.getTheme(theme.id).bg?.toLowerCase();
    if (actual !== theme.surface.toLowerCase()) {
      throw new Error(
        `Code theme "${theme.id}" now paints its canvas ${actual}, but codeThemes.ts ` +
          `and global.css still say ${theme.surface}. Update both.`,
      );
    }
  }
}

/**
 * Remove indentation shared by *every* non-blank line. Purely presentational —
 * relative structure is untouched, and it stops methods lifted out of a class
 * body from rendering with a dead 4-space margin.
 */
function dedent(code: string): string {
  const lines = code.replace(/\s+$/, '').split('\n');
  let min = Infinity;
  for (const line of lines) {
    if (!line.trim()) continue;
    const m = line.match(/^[ \t]*/);
    min = Math.min(min, m ? m[0].length : 0);
    if (min === 0) break;
  }
  if (!isFinite(min) || min === 0) return lines.join('\n');
  return lines.map((l) => (l.trim() ? l.slice(min) : l.trimEnd())).join('\n');
}

export interface HighlightResult {
  /** Shiki `<pre class="shiki">…</pre>` markup, ready for `set:html`. */
  html: string;
  /** The exact text the copy button should put on the clipboard. */
  text: string;
  /** Number of rendered lines, for the line-number gutter. */
  lineCount: number;
}

/** Highlight a snippet at build time. Falls back to plain text for unknown langs. */
export async function highlight(code: string, lang: string): Promise<HighlightResult> {
  const text = dedent(code ?? '');
  const highlighter = await getHighlighter();
  const html = highlighter.codeToHtml(text, {
    lang: SUPPORTED.has(lang) ? lang : 'text',
    themes: THEME_MAP,
    defaultColor: false,
    cssVariablePrefix: CSS_VAR_PREFIX,
    transformers: [dropInheritVars],
  });
  return { html, text, lineCount: text.length ? text.split('\n').length : 1 };
}
