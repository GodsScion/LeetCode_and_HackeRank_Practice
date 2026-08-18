/**
 * Shared data contract for the site.
 *
 * Everything here is derived at BUILD TIME by parsing the solution files in the
 * repo root. The parser is strictly read-only: it never writes to, moves, or
 * reformats any file under "Leet Code/", "Hacker Rank/", or "scratch pad/".
 */

export type Difficulty = 'Easy' | 'Medium' | 'Hard';

export type Lang = 'python' | 'java' | 'javascript' | 'typescript' | 'c';

export type Source = 'leetcode' | 'hackerrank';

/** Display metadata per language. `shiki` is the Shiki grammar id. */
export const LANG_META: Record<Lang, { label: string; shiki: string; ext: string }> = {
  python: { label: 'Python', shiki: 'python', ext: '.py' },
  java: { label: 'Java', shiki: 'java', ext: '.java' },
  javascript: { label: 'JavaScript', shiki: 'javascript', ext: '.js' },
  typescript: { label: 'TypeScript', shiki: 'typescript', ext: '.ts' },
  c: { label: 'C', shiki: 'c', ext: '.c' },
};

/** One language's implementation of a single approach. */
export interface Implementation {
  lang: Lang;
  /** Verbatim source, exactly as it appears in the file. Never reformatted. */
  code: string;
  /** Repo-relative path, e.g. "Leet Code/python.py". */
  sourceFile: string;
  /** 1-indexed line of the header comment that opened this block. */
  startLine: number;
  /** 1-indexed line where the block ends (inclusive). */
  endLine: number;
  /** ISO date this attempt was solved, e.g. "2026-08-17". Absent when undated. */
  attemptedOn?: string;
  /** Verbatim solve time from the header, e.g. "14m". Display only, never parsed. */
  solveTime?: string;
}

/**
 * A distinct way of solving the problem (e.g. "two pointers", "dfs").
 *
 * Approaches are grouped across languages by the optional `[tag]` suffix on the
 * header comment:
 *
 *   # 49. Group Anagrams (https://...) - Medium [sorted-key]
 *   // 49. Group Anagrams (https://...) - Medium [sorted-key]
 *
 * Blocks sharing a tag merge into one approach with multiple language tabs.
 * Untagged blocks each become their own approach, numbered in file order, so
 * the site works with zero annotation and improves as tags are added.
 */
export interface Approach {
  /** Tag slug when tagged, else a stable positional id like "approach-2". */
  id: string;
  /** "Sorted Key" from a tag, else "Approach 2". */
  name: string;
  /** Whether `name` came from an author-written tag or was auto-generated. */
  named: boolean;
  /** Prose from the docstring with the parsed fields below removed. */
  explanation: string;
  /** e.g. "O(n log n)" — parsed from a `Time Complexity:` line. */
  timeComplexity?: string;
  /** e.g. "O(1)" — parsed from a `Space Complexity:` line. */
  spaceComplexity?: string;
  /** Parsed from an optional `Pitfalls:` line. */
  pitfalls?: string;
  /**
   * Reason this approach would not be accepted in an interview, from an optional
   * `Do not use in interview:` line. Present means flagged; the string is the why.
   *
   * These are still worth keeping — they are usually the first thing that came to
   * mind, and knowing why they fail is the lesson. They are sorted to the end of
   * `Problem.approaches` and rendered with a danger label so they can never be
   * mistaken for the answer.
   */
  doNotUseInInterview?: string;
  implementations: Implementation[];
}

export interface Problem {
  source: Source;
  /** LeetCode problem number. Undefined for HackerRank challenges. */
  number?: number;
  /** URL slug for this site, e.g. "49-group-anagrams". */
  slug: string;
  title: string;
  difficulty: Difficulty;
  /** From the `####### SECTION #######` banner above the block. */
  category: string;
  /** Canonical problem URL on LeetCode / HackerRank. */
  url: string;
  /**
   * Derived from the LeetCode slug: https://neetcode.io/solutions/<slug>.
   * Undefined for HackerRank.
   */
  neetcodeUrl?: string;
  /** The author's own restatement. Optional; renders only when present. */
  summary?: string;
  approaches: Approach[];
  /** Union of languages across all approaches, for list-page filtering. */
  languages: Lang[];
  /** Newest `attemptedOn` across every implementation. Undefined when none are dated. */
  lastAttempt?: string;
}

/** A file under "scratch pad/" — rendered read-only with a github.dev edit link. */
export interface ScratchFile {
  /** Repo-relative path, e.g. "scratch pad/python.py". */
  path: string;
  /** Basename, e.g. "python.py". */
  name: string;
  slug: string;
  lang: Lang | 'html';
  code: string;
  lineCount: number;
}

/** The general lessons block at the top of "Leet Code/python.py". */
export interface Note {
  index: number;
  text: string;
}

export interface SiteData {
  problems: Problem[];
  scratch: ScratchFile[];
  notes: Note[];
  /** Build-time counts, surfaced on the home page. */
  stats: {
    problems: number;
    solutions: number;
    byDifficulty: Record<Difficulty, number>;
    byLanguage: Partial<Record<Lang, number>>;
    categories: string[];
  };
}
