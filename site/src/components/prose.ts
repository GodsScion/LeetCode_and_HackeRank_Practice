/**
 * Turn the author's plain-text prose (docstring explanations, notes) into a
 * small, safe subset of HTML.
 *
 * The author's docstrings mix two kinds of newline: real statement breaks
 * ("Refer: …" on its own line) and accidents of hard-wrapping at ~110 columns
 * ("…for condition `x`,\nto make sure…"). They are told apart by the first
 * character of the following line — a line that begins lower-case is a
 * continuation and gets joined, anything else starts a new line. That rule
 * holds across every explanation in the repo.
 *
 * Recognised: blank-line paragraphs, statement breaks, `-`/`*` bullet lists
 * (with continuation lines), `backtick code spans`, and bare http(s) URLs.
 *
 * Everything is escaped before any markup is added.
 */

const ESCAPES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ESCAPES[c]!);
}

const URL_RE = /(https?:\/\/[^\s<>"']+)/g;

function linkify(escaped: string): string {
  return escaped.replace(URL_RE, (raw) => {
    // don't swallow sentence punctuation that trails a URL
    const m = raw.match(/[.,;:!?)\]]+$/);
    const trail = m ? m[0] : '';
    const url = trail ? raw.slice(0, -trail.length) : raw;
    if (!url) return raw;
    return `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>${trail}`;
  });
}

/**
 * The notes cross-reference problems in prose ("inspiration from 19", "look at
 * 143"). When a map of problem number → URL is supplied those become links, but
 * only after an explicit lead-in phrase — a bare number is far too likely to be
 * a complexity bound or an array length.
 */
const XREF_RE = /\b(inspiration from|inspired by|look at|refer to|see|problem)\s+(\d{1,4})\b/gi;

function crossLink(escaped: string, links: Map<number, string>): string {
  return escaped.replace(XREF_RE, (whole, lead: string, digits: string) => {
    const url = links.get(Number(digits));
    return url ? `${lead} <a href="${url}">${digits}</a>` : whole;
  });
}

function inline(text: string, links?: Map<number, string>): string {
  // split on backticks first so URLs inside code spans stay literal
  return escapeHtml(text)
    .split(/`([^`]+)`/g)
    .map((part, i) => {
      if (i % 2 === 1) return `<code>${part}</code>`;
      const linked = linkify(part);
      return links && links.size ? crossLink(linked, links) : linked;
    })
    .join('');
}

const BULLET_RE = /^\s*[-*•]\s+/;
/** A line that opens lower-case is the tail of the line above it. */
const CONTINUATION_RE = /^[a-z]/;

/** Fold hard-wrapped lines back together; keep deliberate statement breaks. */
function foldWrapped(lines: string[]): string[] {
  const out: string[] = [];
  for (const line of lines) {
    if (out.length && CONTINUATION_RE.test(line)) {
      out[out.length - 1] += ` ${line}`;
    } else {
      out.push(line);
    }
  }
  return out;
}

/** Render plain text to HTML. Returns `''` for empty / whitespace-only input. */
export function proseToHtml(
  input: string | undefined | null,
  links?: Map<number, string>,
): string {
  if (!input) return '';
  const text = input.replace(/\r\n?/g, '\n').replace(/[ \t]+$/gm, '').trim();
  if (!text) return '';

  const out: string[] = [];

  for (const block of text.split(/\n{2,}/)) {
    const lines = block.split('\n').filter((l) => l.trim().length > 0);
    if (!lines.length) continue;

    const lead: string[] = [];
    const items: string[] = [];

    for (const line of lines) {
      if (BULLET_RE.test(line)) {
        items.push(line.replace(BULLET_RE, '').trim());
      } else if (items.length) {
        // indented continuation of the previous bullet
        items[items.length - 1] += ` ${line.trim()}`;
      } else {
        lead.push(line.trim());
      }
    }

    if (lead.length) {
      const statements = foldWrapped(lead).map((s) => inline(s, links));
      out.push(`<p>${statements.join('<br />')}</p>`);
    }
    if (items.length) {
      out.push(`<ul>${items.map((i) => `<li>${inline(i, links)}</li>`).join('')}</ul>`);
    }
  }

  return out.join('');
}

/** Cheap plain-text summary for meta descriptions. */
export function toPlain(input: string | undefined | null, max = 160): string {
  if (!input) return '';
  const flat = input.replace(/\s+/g, ' ').replace(/`/g, '').trim();
  return flat.length > max ? `${flat.slice(0, max - 1).trimEnd()}…` : flat;
}
