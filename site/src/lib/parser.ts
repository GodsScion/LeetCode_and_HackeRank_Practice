/**
 * Build-time parser for the practice-solutions site.
 *
 * SAFETY CONTRACT: this module is strictly READ-ONLY over the solution files.
 * It only ever calls readFileSync / readdirSync / statSync. It never writes,
 * moves, reformats or normalises anything under "Leet Code/", "Hacker Rank/",
 * "scratch pad/", "Others/" or "Interviews/". Every emitted `code` string is a
 * verbatim slice of its source file — original indentation, blank lines and
 * comments intact. The only transformation applied to a block is dropping
 * fully-blank leading and trailing lines.
 *
 * Everything is derived by line scanning + small regexes; there is no parser
 * for the underlying languages and there are no dependencies.
 */

import { readFileSync, readdirSync, existsSync, statSync } from 'node:fs';
import { dirname, join, extname, basename } from 'node:path';

import type {
  Approach,
  Difficulty,
  Implementation,
  Lang,
  Note,
  Problem,
  ScratchFile,
  SiteData,
  Source,
} from './types.ts';

/**
 * Repo root, found by walking up until the solution directory appears.
 *
 * A fixed number of `..` hops is not stable across contexts: this module runs from
 * src/lib under `node --experimental-strip-types`, but Astro bundles it into
 * dist/.prerender/chunks for the build, which is one level deeper. Guessing wrong
 * fails silently — every read misses, each file caches as null, and the site builds
 * with zero problems and a successful exit code. Never hardcode an absolute path.
 */
function findRepoRoot(): string {
  let dir = import.meta.dirname ?? process.cwd();
  for (let i = 0; i < 8; i++) {
    if (existsSync(join(dir, 'Leet Code'))) return dir;
    const up = dirname(dir);
    if (up === dir) break;
    dir = up;
  }
  throw new Error(`parser: no "Leet Code" directory above ${import.meta.dirname ?? process.cwd()}`);
}
const REPO_ROOT = findRepoRoot();

/**
 * The LeetCode files, in the order that defines canonical approach ordering:
 * Python first, everything else appends.
 */
const LEETCODE_FILES: { rel: string; lang: Lang }[] = [
  { rel: 'Leet Code/python.py', lang: 'python' },
  { rel: 'Leet Code/java.java', lang: 'java' },
  { rel: 'Leet Code/javascript.js', lang: 'javascript' },
  // The filename really is misspelled in the repo. Do not "fix" it.
  { rel: 'Leet Code/typesript.ts', lang: 'typescript' },
  { rel: 'Leet Code/c.c', lang: 'c' },
];

const HACKERRANK_FILE = 'Hacker Rank/python.py';
const SCRATCH_DIR = 'scratch pad';

const SCRATCH_LANG_BY_EXT: Record<string, Lang | 'html'> = {
  '.py': 'python',
  '.java': 'java',
  '.js': 'javascript',
  '.ts': 'typescript',
  '.c': 'c',
  '.html': 'html',
};

// ---------------------------------------------------------------------------
// File access (read-only)
// ---------------------------------------------------------------------------

const fileCache = new Map<string, string | null>();

function readRepoFile(rel: string): string | null {
  if (!fileCache.has(rel)) {
    const abs = join(REPO_ROOT, rel);
    fileCache.set(rel, existsSync(abs) ? readFileSync(abs, 'utf8') : null);
  }
  return fileCache.get(rel) ?? null;
}

function readRepoLines(rel: string): string[] | null {
  const text = readRepoFile(rel);
  return text === null ? null : text.split('\n');
}

// ---------------------------------------------------------------------------
// Small string helpers
// ---------------------------------------------------------------------------

/** Connecting words that stay lowercase inside a title-cased category. */
const LOWERCASE_WORDS = new Set([
  'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'from', 'in', 'nor', 'of',
  'on', 'or', 'the', 'to', 'vs', 'with',
]);

/** Words that should stay fully uppercase if they ever show up in a banner. */
const ACRONYMS = new Set([
  'BFS', 'DFS', 'DP', 'BST', 'LRU', 'LFU', 'API', 'SQL', 'LIFO', 'FIFO', 'KMP',
]);

function capitalize(word: string): string {
  if (!/[a-z]/i.test(word)) return word; // "/", "&", "1", ...
  if (ACRONYMS.has(word.toUpperCase())) return word.toUpperCase();
  return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
}

/**
 * "ARRAYS AND HASHING" -> "Arrays and Hashing"
 * "1-D DYNAMIC PROGRAMMING" -> "1-D Dynamic Programming"
 * "HEAP / PRIORITY QUEUE" -> "Heap / Priority Queue"
 */
function titleCase(raw: string): string {
  return raw
    .trim()
    .split(/\s+/)
    .map((token, tokenIndex) =>
      token
        .split('-')
        .map((part, partIndex) => {
          const lower = part.toLowerCase();
          const isFirst = tokenIndex === 0 && partIndex === 0;
          if (!isFirst && LOWERCASE_WORDS.has(lower)) return lower;
          return capitalize(part);
        })
        .join('-'),
    )
    .join(' ');
}

/** URL-safe slug fragment: "Two Sum II - Input Array Is Sorted" -> "two-sum-ii-input-array-is-sorted". */
function kebab(raw: string): string {
  return raw
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

/** Inclusive-start / exclusive-end range with fully blank lines trimmed off both ends. */
function trimBlankRange(lines: string[], from: number, to: number): [number, number] {
  let start = from;
  let end = to;
  while (start < end && (lines[start] ?? '').trim() === '') start++;
  while (end > start && (lines[end - 1] ?? '').trim() === '') end--;
  return [start, end];
}

/** Remove the common leading indentation from a set of lines. */
function dedent(lines: string[]): string[] {
  let min = Infinity;
  for (const line of lines) {
    if (line.trim() === '') continue;
    min = Math.min(min, line.length - line.trimStart().length);
  }
  if (!Number.isFinite(min) || min === 0) return lines;
  return lines.map((line) => (line.trim() === '' ? '' : line.slice(min)));
}

// ---------------------------------------------------------------------------
// Header / banner recognition
// ---------------------------------------------------------------------------

/**
 * A problem header, in either comment style:
 *
 *   # 217. Contains Duplicate (https://leetcode.com/problems/...) - Easy
 *   // 49. Group Anagrams (https://...) - Medium [sorted-key]
 *       // 49. ...                                  (Java headers may be indented)
 *
 * The title is lazy so titles containing parentheses still work
 * ("208. Implement Trie (Prefix Tree) (https://...) - Medium"), and the
 * difficulty is optional because a few JS/TS headers omit it — those problems
 * inherit their difficulty from another language's header.
 */
const HEADER_RE =
  /^\s*(?:#+|\/\/)\s*(\d+)\.\s*(.+?)\s*\((https?:\/\/[^)\s]+)\)\s*(?:-\s*(Easy|Medium|Hard)\b)?\s*(.*)$/i;

/**
 * Shorthand header — the problem number and nothing else:
 *
 *   // Problem 424
 *   // 141
 *   // 141 [two-pointer]
 *
 * The line must contain nothing but the number (plus an optional `[tag]`), so a
 * prose comment that merely starts with a digit is never mistaken for a header.
 * Title, URL, difficulty and category are filled in afterwards from the same
 * problem's full header in another file — see resolveInheritedMetadata().
 */
const BARE_HEADER_RE = /^\s*(?:#+|\/\/)\s*(?:problem\s+)?(\d+)\s*(\[[^\]]+\])?\s*$/i;

/**
 * Anything that smells like a problem header. Used purely to report lines that
 * were probably meant to be headers but did not parse, so solutions can never
 * be dropped silently.
 */
const HEADER_HINT_RE = /^\s*(?:#+|\/\/)\s*(?:problem\s+)?\d+\b/i;

/**
 * Section banner in any of the observed shapes — the `#` count and the inner
 * spacing vary, and the JS/Java files wrap the banner in `//`:
 *
 *   ####### ARRAYS AND HASHING #######
 *   ###### 1-D DYNAMIC PROGRAMMING ######
 *   //#######  ARRAYS AND HASHING  #######//
 *   // ##### LINKED LIST ##### //
 */
const BANNER_RE = /^\s*(?:\/\/\s*)?#{3,}\s*([^#]*?)\s*#{3,}\s*(?:\/\/)?\s*$/;

/** `#---------------------- TREE ----------------------#` (HackerRank sub-banner). */
const SUB_BANNER_RE = /^\s*#-{3,}\s*([^-]*?)\s*-{3,}#\s*$/;

/** The trailing scratch/driver section of a file. Everything from here on is not a solution. */
const TAIL_BANNER_RE = /^TEST(?:ING|\s*CASES?)?$/i;

/** `[approach-tag]` suffix on a header. */
const TAG_RE = /\[([^\]]+)\]/;

/** `2026-08-17` anywhere in the header tail. ISO so it sorts as a plain string. */
const DATE_RE = /\b(\d{4}-\d{2}-\d{2})\b/;

/**
 * A `- 14m` / `- 9m 40sec` segment in the header tail.
 *
 * Requires a digit followed by a time unit, which is what stops the existing
 * `- Medium - 2026-01-07 - Duplicate` headers from reporting a solve time of
 * "Duplicate", and stops the date's own `-01-07` from matching. Units are ordered
 * longest-first so `min` never settles for `m`.
 */
const SOLVE_TIME_RE =
  /-\s*(\d+\s*(?:hours|hour|hrs|hr|h|minutes|minute|mins|min|m|seconds|second|secs|sec|s)\b[^-\[]*)/i;

/** Everything but `number` is absent on a shorthand header. */
interface HeaderMatch {
  number: number;
  title?: string;
  url?: string;
  difficulty?: Difficulty;
  tag?: string;
  attemptedOn?: string;
  solveTime?: string;
}

function matchHeader(line: string): HeaderMatch | null {
  const m = HEADER_RE.exec(line);
  if (!m) return null;
  const [, num, title, url, difficulty, rest] = m;
  const tail = rest ?? '';
  const tag = TAG_RE.exec(tail)?.[1]?.trim();
  return {
    number: Number(num),
    title: (title ?? '').trim(),
    url: (url ?? '').trim(),
    difficulty: difficulty ? (titleCase(difficulty) as Difficulty) : undefined,
    tag: tag || undefined,
    attemptedOn: DATE_RE.exec(tail)?.[1],
    solveTime: SOLVE_TIME_RE.exec(tail)?.[1]?.trim(),
  };
}

function matchBareHeader(line: string): HeaderMatch | null {
  const m = BARE_HEADER_RE.exec(line);
  if (!m) return null;
  const tag = m[2] ? TAG_RE.exec(m[2])?.[1]?.trim() : undefined;
  return { number: Number(m[1]), tag: tag || undefined };
}

function matchBanner(line: string): string | null {
  const m = BANNER_RE.exec(line);
  const text = m?.[1]?.trim();
  return text ? text : null;
}

/**
 * Section banners differ slightly between the LeetCode and HackerRank files
 * ("TREE" vs "TREES"), and the category nav would then show near-duplicate
 * sections. Fold the known variants onto one canonical name. Deliberately an
 * explicit list rather than a pluralisation rule — extend it when a new banner
 * shows up.
 */
const CATEGORY_ALIASES: Record<string, string> = {
  Tree: 'Trees',
  Trie: 'Tries',
  Heap: 'Heap / Priority Queue',
  Graph: 'Graphs',
  'Linked Lists': 'Linked List',
};

function canonicalCategory(banner: string): string {
  const name = titleCase(banner);
  return CATEGORY_ALIASES[name] ?? name;
}

// ---------------------------------------------------------------------------
// Docstring metadata (Python)
// ---------------------------------------------------------------------------

const DOCSTRING_OPEN_RE = /^[ \t]*('''|""")/m;
const TIME_RE = /^\s*-?\s*time\s+complexity\s*:\s*(.+)$/i;
const SPACE_RE = /^\s*-?\s*space\s+complexity\s*:\s*(.+)$/i;
const PITFALL_RE = /^\s*-?\s*pitfalls?\s*:\s*(.+)$/i;

/**
 * `Do not use in interview: <reason>` — marks an approach that is accepted by the
 * judge but would not pass a human. Also accepts "Don't"/"Do not use in an
 * interview" so the line can be written the way it is said out loud.
 */
const NO_INTERVIEW_RE = /^\s*-?\s*do\s*n[o']?t\s+use\s+in\s+(?:an\s+)?interviews?\s*:\s*(.+)$/i;

interface DocMeta {
  explanation: string;
  timeComplexity?: string;
  spaceComplexity?: string;
  pitfalls?: string;
  doNotUseInInterview?: string;
}

/** Sphinx field lines that LeetCode's starter docstrings are made of. */
const SPHINX_FIELD_RE = /^:(?:type|rtype|param|returns?|raises|yields?)\b/i;

/** Verbatim phrasing from LeetCode's own starter code. */
const LEETCODE_STUB_LINES = [
  /^do not return anything, modify .+ in-?place instead\.?$/i,
  /^encodes? a tree to a single string\.?$/i,
  /^decodes? your encoded data to tree\.?$/i,
  /^initialize your data structure here\.?$/i,
  /^your \w+ object will be instantiated and called as such:?$/i,
];

/**
 * True when a docstring is nothing but LeetCode's own starter text. That text
 * is theirs, not the author's, and this site reproduces none of their problem
 * content — so it yields no explanation at all.
 *
 * Deliberately conservative: one line of the author's own prose anywhere in the
 * docstring disqualifies the whole thing. A false positive silently deletes
 * their writing, which is far worse than one stub slipping through.
 */
function isBoilerplateDocstring(explanation: string): boolean {
  const lines = explanation
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) return false;
  return lines.every(
    (line) => SPHINX_FIELD_RE.test(line) || LEETCODE_STUB_LINES.some((re) => re.test(line)),
  );
}

/** Trim decoration off a parsed value: trailing punctuation and wrapping backticks. */
function cleanValue(raw: string): string {
  let value = raw.trim().replace(/[.,;]+$/, '').trim();
  if (value.length > 1 && value.startsWith('`') && value.endsWith('`')) {
    value = value.slice(1, -1).trim();
  }
  return value;
}

/**
 * Pull Time/Space complexity and Pitfalls out of the first triple-quoted block
 * in `code`, leaving the remaining prose as the explanation.
 *
 * NOTE: the docstring is deliberately left in `code` — this only affects the
 * derived metadata, never the source we render.
 */
function parseDocMeta(code: string): DocMeta {
  const open = DOCSTRING_OPEN_RE.exec(code);
  if (!open) return { explanation: '' };

  const quote = open[1]!;
  const bodyStart = open.index + open[0].length;
  const bodyEnd = code.indexOf(quote, bodyStart);
  const body = bodyEnd === -1 ? code.slice(bodyStart) : code.slice(bodyStart, bodyEnd);

  const meta: DocMeta = { explanation: '' };
  const prose: string[] = [];

  // Text sitting on the opening-quote line carries no indentation of its own,
  // so it is normalised separately and the common indent is measured from the
  // remaining lines (same rule as PEP 257).
  const raw = body.split('\n');
  const normalised = [(raw[0] ?? '').trim(), ...dedent(raw.slice(1))];

  for (const line of normalised) {
    const time = TIME_RE.exec(line);
    if (time && meta.timeComplexity === undefined) {
      meta.timeComplexity = cleanValue(time[1]!);
      continue;
    }
    const space = SPACE_RE.exec(line);
    if (space && meta.spaceComplexity === undefined) {
      meta.spaceComplexity = cleanValue(space[1]!);
      continue;
    }
    const pitfall = PITFALL_RE.exec(line);
    if (pitfall && meta.pitfalls === undefined) {
      meta.pitfalls = cleanValue(pitfall[1]!);
      continue;
    }
    const noInterview = NO_INTERVIEW_RE.exec(line);
    if (noInterview && meta.doNotUseInInterview === undefined) {
      meta.doNotUseInInterview = cleanValue(noInterview[1]!);
      continue;
    }
    prose.push(line);
  }

  const [from, to] = trimBlankRange(prose, 0, prose.length);
  const explanation = prose.slice(from, to).join('\n').trim();
  meta.explanation = isBoilerplateDocstring(explanation) ? '' : explanation;
  return meta;
}

// ---------------------------------------------------------------------------
// Brace tracking (Java / JS / TS)
// ---------------------------------------------------------------------------

/**
 * Blank out char literals, string literals and comments so brace counting is
 * not thrown off by things like `case '}':` in the Valid Parentheses solution.
 */
function stripBraceNoise(line: string): string {
  return line
    .replace(/'(?:\\.|[^'\\])'/g, "'x'")
    .replace(/"(?:\\.|[^"\\])*"/g, '""')
    .replace(/\/\*.*?\*\//g, '')
    .replace(/\/\/.*$/, '')
    .replace(/^\s*\*.*$/, ''); // javadoc continuation line
}

// ---------------------------------------------------------------------------
// Block scanning
// ---------------------------------------------------------------------------

/** One parsed solution block, before problems are merged together. */
interface RawBlock {
  source: Source;
  number?: number;
  title?: string;
  url?: string;
  difficulty?: Difficulty;
  category?: string;
  tag?: string;
  attemptedOn?: string;
  solveTime?: string;
  lang: Lang;
  code: string;
  sourceFile: string;
  startLine: number;
  endLine: number;
  explanation: string;
  timeComplexity?: string;
  spaceComplexity?: string;
  pitfalls?: string;
  doNotUseInInterview?: string;
  /** The source line that opened this block, for diagnostics. */
  headerText?: string;
}

/**
 * Scan one single-file LeetCode source. A block starts at a problem header and
 * runs until the next header, the next section banner, the end of the enclosing
 * class (brace languages), a trailing test/driver banner, or EOF.
 */
function scanLeetCodeFile(rel: string, lang: Lang, unparsed: string[]): RawBlock[] {
  const lines = readRepoLines(rel);
  if (!lines) return [];

  const tracksBraces = lang !== 'python';
  const blocks: RawBlock[] = [];

  let category: string | undefined;
  let header: HeaderMatch | null = null;
  let headerIndex = -1;
  let headerCategory: string | undefined;
  let depth = 0;

  /** Close the open block; `endExclusive` is the index of the line that ended it. */
  const closeBlock = (endExclusive: number): void => {
    if (!header) return;
    const [from, to] = trimBlankRange(lines, headerIndex + 1, endExclusive);
    if (to > from) {
      const code = lines.slice(from, to).join('\n');
      const meta = lang === 'python' ? parseDocMeta(code) : { explanation: '' };
      blocks.push({
        source: 'leetcode',
        number: header.number,
        title: header.title,
        url: header.url,
        difficulty: header.difficulty,
        category: headerCategory,
        tag: header.tag,
        attemptedOn: header.attemptedOn,
        solveTime: header.solveTime,
        lang,
        code,
        sourceFile: rel,
        startLine: headerIndex + 1,
        endLine: to,
        headerText: lines[headerIndex]?.trim(),
        ...meta,
      });
    } else {
      unparsed.push(`${rel}:${headerIndex + 1}: header has an empty body — ${lines[headerIndex]?.trim()}`);
    }
    header = null;
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;

    // Leaving the enclosing class ends both the block and the useful part of
    // the file (java.java is followed by a `class Main` driver and dead code).
    if (header && tracksBraces) {
      let escaped = false;
      for (const ch of stripBraceNoise(line)) {
        if (ch === '{') depth++;
        else if (ch === '}' && --depth < 0) {
          escaped = true;
          break;
        }
      }
      if (escaped) {
        closeBlock(i);
        break;
      }
    }

    const banner = matchBanner(line);
    if (banner) {
      closeBlock(i);
      if (TAIL_BANNER_RE.test(banner)) break;
      category = canonicalCategory(banner);
      continue;
    }

    // Full header first, then the shorthand form, which is the stricter match.
    const next = matchHeader(line) ?? matchBareHeader(line);
    if (next) {
      closeBlock(i);
      header = next;
      headerIndex = i;
      headerCategory = category;
      depth = 0;
      continue;
    }

    if (HEADER_HINT_RE.test(line) || /leetcode\.com\/problems\//.test(line)) {
      unparsed.push(`${rel}:${i + 1}: ${line.trim()}`);
    }
  }
  closeBlock(lines.length);

  return blocks;
}

// ---------------------------------------------------------------------------
// HackerRank
// ---------------------------------------------------------------------------

const HR_OPEN_RE =
  /^\s*#<<\s*(.+?)\s*\((https?:\/\/[^)\s]+)\)\s*(?:-\s*(Easy|Medium|Hard)\b)?\s*(.*)$/i;
const HR_CLOSE_RE = /^\s*#>>\s*$/;
const HR_STUB_OPEN_RE = /^\s*#<\s*Stub Code\s*$/i;
const HR_STUB_CLOSE_RE = /^\s*#>\s*Stub Code\s*$/i;

/**
 * Scan "Hacker Rank/python.py".
 *
 * Problems open with `#<< Title (url) - Difficulty` (titles may contain colons
 * and brackets) and close with `#>>`. Regions fenced by `#< Stub Code` /
 * `#> Stub Code` are HackerRank's own boilerplate and are excluded from the
 * emitted code; there can be several per problem.
 */
function scanHackerRank(unparsed: string[]): RawBlock[] {
  const rel = HACKERRANK_FILE;
  const lines = readRepoLines(rel);
  if (!lines) return [];

  const blocks: RawBlock[] = [];
  let section: string | undefined;
  let subSection: string | undefined;

  let open: { title: string; url: string; difficulty?: Difficulty; tag?: string } | null = null;
  let openIndex = -1;
  let openCategory: string | undefined;
  let body: { line: string; lineNumber: number }[] = [];
  let inStub = false;

  const closeProblem = (): void => {
    if (!open) return;
    const text = body.map((b) => b.line);
    const [from, to] = trimBlankRange(text, 0, text.length);
    if (to > from) {
      const code = text.slice(from, to).join('\n');
      blocks.push({
        source: 'hackerrank',
        number: undefined,
        title: open.title,
        url: open.url,
        difficulty: open.difficulty,
        category: openCategory,
        tag: open.tag,
        lang: 'python',
        code,
        sourceFile: rel,
        startLine: openIndex + 1,
        endLine: body[to - 1]!.lineNumber,
        ...parseDocMeta(code),
      });
    } else {
      unparsed.push(`${rel}:${openIndex + 1}: problem has no code outside its stub regions`);
    }
    open = null;
    body = [];
    inStub = false;
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;

    if (open) {
      if (HR_STUB_OPEN_RE.test(line)) {
        inStub = true;
        continue;
      }
      if (HR_STUB_CLOSE_RE.test(line)) {
        inStub = false;
        continue;
      }
      if (HR_CLOSE_RE.test(line)) {
        closeProblem();
        continue;
      }
      if (!inStub) body.push({ line, lineNumber: i + 1 });
      continue;
    }

    const sub = SUB_BANNER_RE.exec(line)?.[1]?.trim();
    if (sub) {
      subSection = canonicalCategory(sub);
      continue;
    }

    const banner = matchBanner(line);
    if (banner) {
      section = canonicalCategory(banner);
      subSection = undefined;
      continue;
    }

    const m = HR_OPEN_RE.exec(line);
    if (m) {
      const [, title, url, difficulty, rest] = m;
      const tag = TAG_RE.exec(rest ?? '')?.[1]?.trim();
      open = {
        title: (title ?? '').trim(),
        url: (url ?? '').trim(),
        difficulty: difficulty ? (titleCase(difficulty) as Difficulty) : undefined,
        tag: tag || undefined,
      };
      openIndex = i;
      // The most specific banner wins.
      openCategory = subSection ?? section;
      continue;
    }

    if (/^\s*#<</.test(line)) unparsed.push(`${rel}:${i + 1}: ${line.trim()}`);
  }
  closeProblem(); // unterminated final problem, if any

  return blocks;
}

// ---------------------------------------------------------------------------
// Notes
// ---------------------------------------------------------------------------

/**
 * The `'''NOTES: ... '''` block at the top of "Leet Code/python.py". Items are
 * numbered `1.` onward; wrapped lines are re-joined, bullets and blank
 * lines are kept as structure.
 */
function parseNotes(): Note[] {
  const text = readRepoFile('Leet Code/python.py');
  if (!text) return [];

  const start = text.indexOf("'''");
  if (start === -1) return [];
  const end = text.indexOf("'''", start + 3);
  const block = end === -1 ? text.slice(start + 3) : text.slice(start + 3, end);
  if (!/^\s*NOTES:/m.test(block)) return [];

  const notes: Note[] = [];
  let current: { index: number; lines: string[] } | null = null;

  for (const line of block.split('\n')) {
    const m = /^\s*(\d+)\.\s+(.*)$/.exec(line);
    if (m) {
      if (current) notes.push({ index: current.index, text: collapseNote(current.lines) });
      current = { index: Number(m[1]), lines: [m[2] ?? ''] };
    } else if (current) {
      current.lines.push(line);
    }
  }
  if (current) notes.push({ index: current.index, text: collapseNote(current.lines) });

  return notes;
}

function collapseNote(raw: string[]): string {
  const out: string[] = [];
  let buffer = '';
  const flush = (): void => {
    if (buffer.trim()) out.push(buffer.trim());
    buffer = '';
  };

  for (const line of raw) {
    const text = line.trim().replace(/\s+/g, ' ');
    if (text === '') {
      flush();
      out.push('');
      continue;
    }
    if (/^[-*\u2022]\s/.test(text)) {
      flush();
      buffer = text;
      continue;
    }
    buffer = buffer ? `${buffer} ${text}` : text;
  }
  flush();

  return out.join('\n').replace(/\n{3,}/g, '\n\n').trim();
}

// ---------------------------------------------------------------------------
// Scratch pad
// ---------------------------------------------------------------------------

function parseScratch(): ScratchFile[] {
  const dirAbs = join(REPO_ROOT, SCRATCH_DIR);
  if (!existsSync(dirAbs) || !statSync(dirAbs).isDirectory()) return [];

  const files: ScratchFile[] = [];
  for (const name of readdirSync(dirAbs).sort()) {
    const lang = SCRATCH_LANG_BY_EXT[extname(name).toLowerCase()];
    if (!lang) continue;
    const rel = `${SCRATCH_DIR}/${name}`;
    const lines = readRepoLines(rel);
    if (!lines) continue;
    const [from, to] = trimBlankRange(lines, 0, lines.length);
    const code = lines.slice(from, to).join('\n');
    files.push({
      path: rel,
      name: basename(name),
      slug: kebab(name),
      lang,
      code,
      lineCount: to - from,
    });
  }
  return files;
}

// ---------------------------------------------------------------------------
// Merging blocks into problems
// ---------------------------------------------------------------------------

function neetcodeUrlFor(url: string): string | undefined {
  const m = /leetcode\.com\/problems\/([^/?#]+)/.exec(url);
  return m ? `https://neetcode.io/solutions/${m[1]}` : undefined;
}

/**
 * Second pass over every block from every file.
 *
 * Shorthand headers (`// 141`) and header-less standalone files carry only a
 * problem number, so their title / URL / difficulty / category are borrowed
 * from the first full header for that same number. This runs over all files at
 * once, so it does not matter which file the number was first spelled out in.
 *
 * A block whose number matches no full header anywhere cannot be placed on the
 * site at all — it is reported rather than emitted half-built.
 */
function resolveInheritedMetadata(blocks: RawBlock[], unparsed: string[]): RawBlock[] {
  type Known = { title?: string; url?: string; difficulty?: Difficulty; category?: string };
  const known = new Map<number, Known>();

  for (const block of blocks) {
    if (block.number === undefined) continue;
    let entry = known.get(block.number);
    if (!entry) {
      entry = {};
      known.set(block.number, entry);
    }
    entry.title ??= block.title;
    entry.url ??= block.url;
    entry.difficulty ??= block.difficulty;
    entry.category ??= block.category;
  }

  const resolved: RawBlock[] = [];
  for (const block of blocks) {
    if (block.url) {
      resolved.push(block);
      continue;
    }
    const reference = block.number === undefined ? undefined : known.get(block.number);
    if (!reference?.url) {
      unparsed.push(
        `${block.sourceFile}:${block.startLine}: ${block.headerText ?? 'solution block'} — no full header for problem ${block.number ?? '?'} in any file, so it cannot be placed`,
      );
      continue;
    }
    resolved.push({
      ...block,
      title: block.title ?? reference.title,
      url: reference.url,
      difficulty: block.difficulty ?? reference.difficulty,
      category: block.category ?? reference.category,
    });
  }
  return resolved;
}

function buildProblems(blocks: RawBlock[]): Problem[] {
  // Group by problem, preserving first-seen order (Python file order first).
  const order: string[] = [];
  const grouped = new Map<string, RawBlock[]>();
  for (const block of blocks) {
    const key =
      block.source === 'leetcode' ? `lc:${block.number}` : `hr:${kebab(block.title ?? '')}`;
    let bucket = grouped.get(key);
    if (!bucket) {
      bucket = [];
      grouped.set(key, bucket);
      order.push(key);
    }
    bucket.push(block);
  }

  const problems: Problem[] = [];
  for (const key of order) {
    const bucket = grouped.get(key)!;
    const first = bucket[0]!;

    const approaches = buildApproaches(bucket);
    const languages: Lang[] = [];
    for (const approach of approaches) {
      for (const impl of approach.implementations) {
        if (!languages.includes(impl.lang)) languages.push(impl.lang);
      }
    }

    const title = bucket.find((b) => b.title)?.title ?? '';
    const url = bucket.find((b) => b.url)?.url ?? '';
    const difficulty = bucket.find((b) => b.difficulty)?.difficulty;
    const category = bucket.find((b) => b.category)?.category ?? '';

    problems.push({
      source: first.source,
      number: first.source === 'leetcode' ? first.number : undefined,
      slug:
        first.source === 'leetcode'
          ? `${first.number}-${kebab(title)}`.replace(/-+$/, '')
          : `hr-${kebab(title)}`,
      title,
      // Cast: types.ts requires a Difficulty. Every problem in the repo today
      // has one on at least one header; if that ever stops being true this
      // lands as undefined and verify-parse.ts fails loudly rather than the
      // parser inventing a difficulty.
      difficulty: difficulty as Difficulty,
      category,
      url,
      neetcodeUrl: first.source === 'leetcode' ? neetcodeUrlFor(url) : undefined,
      // No `summary:` convention exists in the source files yet.
      summary: undefined,
      approaches,
      languages,
      lastAttempt: approaches
        .flatMap((a) => a.implementations)
        .map((i) => i.attemptedOn)
        .filter((d): d is string => !!d)
        .sort()
        .at(-1),
    });
  }

  return problems;
}

/**
 * Blocks sharing a `[tag]` merge into one approach with several language tabs.
 * Untagged blocks each become their own approach, numbered by position.
 */
function buildApproaches(blocks: RawBlock[]): Approach[] {
  const approaches: Approach[] = [];
  const byTag = new Map<string, Approach>();

  for (const block of blocks) {
    const implementation: Implementation = {
      lang: block.lang,
      code: block.code,
      sourceFile: block.sourceFile,
      startLine: block.startLine,
      endLine: block.endLine,
      attemptedOn: block.attemptedOn,
      solveTime: block.solveTime,
    };

    const tagId = block.tag ? kebab(block.tag) : '';
    if (tagId) {
      let approach = byTag.get(tagId);
      if (!approach) {
        approach = {
          id: tagId,
          name: titleCase(tagId.replace(/-/g, ' ')),
          named: true,
          explanation: '',
          implementations: [],
        };
        byTag.set(tagId, approach);
        approaches.push(approach);
      }
      approach.implementations.push(implementation);
      if (!approach.explanation) approach.explanation = block.explanation;
      approach.timeComplexity ??= block.timeComplexity;
      approach.spaceComplexity ??= block.spaceComplexity;
      approach.pitfalls ??= block.pitfalls;
      approach.doNotUseInInterview ??= block.doNotUseInInterview;
    } else {
      approaches.push({
        id: '',
        name: '',
        named: false,
        explanation: block.explanation,
        timeComplexity: block.timeComplexity,
        spaceComplexity: block.spaceComplexity,
        pitfalls: block.pitfalls,
        doNotUseInInterview: block.doNotUseInInterview,
        implementations: [implementation],
      });
    }
  }

  // Anything flagged `Do not use in interview:` sinks to the bottom, keeping its
  // relative order. Sorting here rather than in the page means the section order,
  // the sidebar list and the "Solution N" numbering can never disagree.
  const ordered = [
    ...approaches.filter((a) => !a.doNotUseInInterview),
    ...approaches.filter((a) => a.doNotUseInInterview),
  ];

  // Positional ids/names for the untagged ones, then guarantee uniqueness.
  const used = new Set<string>();
  ordered.forEach((approach, i) => {
    if (!approach.named) {
      approach.id = `approach-${i + 1}`;
      approach.name = `Approach ${i + 1}`;
    }
    let id = approach.id;
    for (let n = 2; used.has(id); n++) id = `${approach.id}-${n}`;
    approach.id = id;
    used.add(id);
  });

  return ordered;
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

export interface ParseDiagnostics {
  /**
   * Lines that looked like a problem header (or a header-less solution file)
   * but did not parse. Never empty-by-design: this is the guard against
   * silently dropping a solution.
   */
  unparsedHeaders: string[];
}

/** `parseAll()` plus the non-fatal diagnostics that verify-parse.ts reports on. */
export function parseAllWithDiagnostics(): { data: SiteData; diagnostics: ParseDiagnostics } {
  const unparsedHeaders: string[] = [];

  const blocks: RawBlock[] = [];
  for (const { rel, lang } of LEETCODE_FILES) {
    blocks.push(...scanLeetCodeFile(rel, lang, unparsedHeaders));
  }
  blocks.push(...scanHackerRank(unparsedHeaders));

  const problems = buildProblems(resolveInheritedMetadata(blocks, unparsedHeaders));
  const scratch = parseScratch();
  const notes = parseNotes();

  const byDifficulty: Record<Difficulty, number> = { Easy: 0, Medium: 0, Hard: 0 };
  const byLanguage: Partial<Record<Lang, number>> = {};
  const categories: string[] = [];
  let solutions = 0;

  for (const problem of problems) {
    if (problem.difficulty in byDifficulty) byDifficulty[problem.difficulty]++;
    if (problem.category && !categories.includes(problem.category)) {
      categories.push(problem.category);
    }
    for (const approach of problem.approaches) {
      for (const impl of approach.implementations) {
        solutions++;
        byLanguage[impl.lang] = (byLanguage[impl.lang] ?? 0) + 1;
      }
    }
  }

  return {
    data: {
      problems,
      scratch,
      notes,
      stats: {
        problems: problems.length,
        solutions,
        byDifficulty,
        byLanguage,
        categories,
      },
    },
    diagnostics: { unparsedHeaders },
  };
}

export function parseAll(): SiteData {
  return parseAllWithDiagnostics().data;
}
