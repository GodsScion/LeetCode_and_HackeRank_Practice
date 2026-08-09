/**
 * Small build-time helpers shared by pages and components.
 *
 * NOTE: everything internal must be routed through `href()`. The site is served
 * from the GitHub Pages sub-path `/Syntax-Shenanigans/`, so any
 * hard-coded root-absolute URL 404s in production.
 */

export const REPO = 'GodsScion/Syntax-Shenanigans';
export const REPO_URL = `https://github.com/${REPO}`;
export const BRANCH = 'main';
export const OWNER = 'Sai Vignesh Golla';

const BASE = import.meta.env.BASE_URL;

/** Prefix an internal path with the configured base. `href('/notes/')`. */
export function href(path: string): string {
  const base = BASE.endsWith('/') ? BASE.slice(0, -1) : BASE;
  const rest = path.startsWith('/') ? path : `/${path}`;
  return `${base}${rest}` || '/';
}

/** Encode each path segment so `Leet Code/python.py` survives as a URL. */
function encodePath(repoPath: string): string {
  return repoPath.split('/').map(encodeURIComponent).join('/');
}

/** Permalink to a line range in the repo on github.com. */
export function githubBlobUrl(
  repoPath: string,
  startLine?: number,
  endLine?: number,
): string {
  const base = `${REPO_URL}/blob/${BRANCH}/${encodePath(repoPath)}`;
  if (!startLine) return base;
  return endLine && endLine !== startLine
    ? `${base}#L${startLine}-L${endLine}`
    : `${base}#L${startLine}`;
}

/** Open the file straight into the github.dev web editor. */
export function githubDevUrl(repoPath: string): string {
  return `https://github.dev/${REPO}/blob/${BRANCH}/${encodePath(repoPath)}`;
}

/** github.dev, opened on a directory rather than a single file. */
export function githubDevTreeUrl(repoPath = ''): string {
  return repoPath
    ? `https://github.dev/${REPO}/tree/${BRANCH}/${encodePath(repoPath)}`
    : `https://github.dev/${REPO}`;
}

/** Labels for every grammar the site can render, including scratch-only HTML. */
export const LANG_LABEL: Record<string, string> = {
  python: 'Python',
  java: 'Java',
  javascript: 'JavaScript',
  typescript: 'TypeScript',
  c: 'C',
  html: 'HTML',
};

/** Display label for a problem's origin. */
export const SOURCE_LABEL: Record<string, string> = {
  leetcode: 'LeetCode',
  hackerrank: 'HackerRank',
};

/** `#217` for LeetCode, a source tag for numberless HackerRank challenges. */
export function problemRef(p: { number?: number; source: string }): string {
  if (typeof p.number === 'number') return `#${p.number}`;
  return p.source === 'hackerrank' ? 'HR' : '—';
}
