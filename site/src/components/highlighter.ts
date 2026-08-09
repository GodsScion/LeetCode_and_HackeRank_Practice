/**
 * ONE Shiki highlighter for the whole build.
 *
 * `createHighlighter()` loads WASM + grammars + theme, which costs ~1s. This
 * module is evaluated once per Vite module graph, so every page reuses the same
 * instance and the same in-flight promise. Never call `codeToHtml` from `shiki`
 * directly — that spins up a fresh highlighter per call.
 *
 * Theme: `vitesse-dark`. Of the three candidates it is the only one whose
 * canvas (#121212) and palette are quiet enough to sit on #050505 without
 * fighting it — `github-dark-default` reads blue-cool and branded next to the
 * warm neutrals here, and `min-dark` throws away too much token colour to be
 * useful as a daily reference. Vitesse's desaturated sage/terracotta set also
 * happens to harmonise with the difficulty palette.
 */
import { createHighlighter, type Highlighter } from 'shiki';

export const CODE_THEME = 'vitesse-dark';

/** Only the grammars this repo actually contains. */
const LANGS = ['python', 'java', 'javascript', 'typescript', 'c', 'html'] as const;

const SUPPORTED = new Set<string>(LANGS);

let instance: Promise<Highlighter> | undefined;

function getHighlighter(): Promise<Highlighter> {
  if (!instance) {
    instance = createHighlighter({
      themes: [CODE_THEME],
      langs: [...LANGS],
    });
  }
  return instance;
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
    theme: CODE_THEME,
  });
  return { html, text, lineCount: text.length ? text.split('\n').length : 1 };
}
