/**
 * verify-parse — sanity check the build-time parser.
 *
 *   cd site && node --experimental-strip-types scripts/verify-parse.ts
 *   cd site && npm run verify
 *
 * Prints the counts the site is built from, then hard-fails (exit 1) on
 * anything that would render as a broken page or a silently dropped solution.
 * It is read-only: it calls parseAll() and inspects the result, nothing else.
 */

import { parseAllWithDiagnostics } from '../src/lib/parser.ts';
import type { Difficulty, Lang, Problem } from '../src/lib/types.ts';

const { data, diagnostics } = parseAllWithDiagnostics();
const { problems, scratch, notes, stats } = data;

const failures: string[] = [];
const fail = (message: string): void => {
  failures.push(message);
};

// ---------------------------------------------------------------------------
// Counts
// ---------------------------------------------------------------------------

let approachCount = 0;
let implementationCount = 0;
const bySource: Record<string, number> = {};
const byDifficulty: Record<string, number> = { Easy: 0, Medium: 0, Hard: 0 };
const byLanguage: Partial<Record<Lang, number>> = {};

for (const problem of problems) {
  bySource[problem.source] = (bySource[problem.source] ?? 0) + 1;
  if (problem.difficulty) byDifficulty[problem.difficulty] = (byDifficulty[problem.difficulty] ?? 0) + 1;
  for (const approach of problem.approaches) {
    approachCount++;
    for (const impl of approach.implementations) {
      implementationCount++;
      byLanguage[impl.lang] = (byLanguage[impl.lang] ?? 0) + 1;
    }
  }
}

const pairs = (record: Record<string, number | undefined>): string =>
  Object.entries(record)
    .filter(([, n]) => n)
    .map(([k, n]) => `${k} ${n}`)
    .join(', ') || '(none)';

console.log('=== parse report ===');
console.log(`Problems         ${problems.length}`);
console.log(`Approaches       ${approachCount}`);
console.log(`Implementations  ${implementationCount}`);
console.log(`By source        ${pairs(bySource)}`);
console.log(`By language      ${pairs(byLanguage)}`);
console.log(`By difficulty    ${pairs(byDifficulty)}`);
console.log(`Categories (${stats.categories.length})   ${stats.categories.join(', ')}`);
console.log(`Notes            ${notes.length}`);
console.log(`Scratch files    ${scratch.length}  (${scratch.map((s) => s.name).join(', ')})`);

// ---------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------

const DIFFICULTIES: Difficulty[] = ['Easy', 'Medium', 'Hard'];

if (problems.length === 0) fail('no problems were parsed at all');

const where = (p: Problem): string =>
  `${p.slug || '(no slug)'} [${p.approaches[0]?.implementations[0]?.sourceFile ?? '?'}:${
    p.approaches[0]?.implementations[0]?.startLine ?? '?'
  }]`;

const seenSlugs = new Map<string, string>();
for (const problem of problems) {
  const label = where(problem);

  if (!problem.slug) fail(`${label}: empty slug`);
  else if (!/^[a-z0-9][a-z0-9-]*$/.test(problem.slug)) fail(`${label}: slug is not URL-safe`);
  else if (seenSlugs.has(problem.slug)) fail(`duplicate slug "${problem.slug}" (also ${seenSlugs.get(problem.slug)})`);
  else seenSlugs.set(problem.slug, label);

  if (!problem.title?.trim()) fail(`${label}: missing title`);
  if (!DIFFICULTIES.includes(problem.difficulty)) fail(`${label}: missing/invalid difficulty (${problem.difficulty})`);
  if (!problem.category?.trim()) fail(`${label}: missing category`);

  if (!problem.url) {
    fail(`${label}: missing url`);
  } else if (problem.source === 'leetcode') {
    if (!/^https:\/\/leetcode\.com\/problems\/[a-z0-9-]+\/?/.test(problem.url)) {
      fail(`${label}: malformed LeetCode url ${problem.url}`);
    }
    if (!problem.neetcodeUrl) fail(`${label}: missing neetcodeUrl`);
    else if (!/^https:\/\/neetcode\.io\/solutions\/[a-z0-9-]+$/.test(problem.neetcodeUrl)) {
      fail(`${label}: malformed neetcodeUrl ${problem.neetcodeUrl}`);
    }
    if (typeof problem.number !== 'number' || !Number.isFinite(problem.number)) {
      fail(`${label}: LeetCode problem without a number`);
    }
  } else {
    if (!/^https:\/\/www\.hackerrank\.com\/challenges\/[a-z0-9-]+\//.test(problem.url)) {
      fail(`${label}: malformed HackerRank url ${problem.url}`);
    }
    if (problem.neetcodeUrl) fail(`${label}: HackerRank problem must not have a neetcodeUrl`);
    if (problem.number !== undefined) fail(`${label}: HackerRank problem must not have a number`);
  }

  if (problem.approaches.length === 0) fail(`${label}: zero approaches`);

  const approachIds = new Set<string>();
  for (const approach of problem.approaches) {
    if (!approach.id) fail(`${label}: approach with an empty id`);
    else if (approachIds.has(approach.id)) fail(`${label}: duplicate approach id "${approach.id}"`);
    else approachIds.add(approach.id);

    if (!approach.name) fail(`${label}: approach "${approach.id}" has no name`);
    if (approach.implementations.length === 0) fail(`${label}: approach "${approach.id}" has zero implementations`);

    for (const impl of approach.implementations) {
      if (!impl.code.trim()) {
        fail(`${label}: empty code block (${impl.sourceFile}:${impl.startLine})`);
      }
      if (impl.startLine < 1 || impl.endLine < impl.startLine) {
        fail(`${label}: bad line range ${impl.sourceFile}:${impl.startLine}-${impl.endLine}`);
      }
      if (!problem.languages.includes(impl.lang)) {
        fail(`${label}: languages[] is missing "${impl.lang}"`);
      }
    }
  }
}

const scratchSlugs = new Set<string>();
for (const file of scratch) {
  if (!file.code.trim()) fail(`scratch ${file.path}: empty`);
  if (scratchSlugs.has(file.slug)) fail(`duplicate scratch slug "${file.slug}"`);
  scratchSlugs.add(file.slug);
}

for (const note of notes) {
  if (!note.text.trim()) fail(`note ${note.index}: empty text`);
}

if (stats.problems !== problems.length) fail('stats.problems disagrees with problems.length');
if (stats.solutions !== implementationCount) fail('stats.solutions disagrees with the implementation count');

// ---------------------------------------------------------------------------
// Warnings (never fatal)
// ---------------------------------------------------------------------------

const missingComplexity = problems.filter((p) =>
  p.approaches.every((a) => !a.timeComplexity && !a.spaceComplexity),
);
const partialComplexity = problems.filter(
  (p) =>
    !missingComplexity.includes(p) &&
    p.approaches.some((a) => !a.timeComplexity || !a.spaceComplexity),
);

console.log(
  `\nComplexity coverage: ${problems.length - missingComplexity.length}/${problems.length} problems have it on at least one approach.`,
);
if (missingComplexity.length) {
  console.log(`WARNING — ${missingComplexity.length} problem(s) with no complexity anywhere:`);
  for (const p of missingComplexity) {
    console.log(`  - ${p.slug}  (${p.category}, ${p.difficulty})`);
  }
}
if (partialComplexity.length) {
  console.log(`WARNING — ${partialComplexity.length} problem(s) where some approaches are missing complexity.`);
}

// Not a warning: these are deliberate. Listing them is how you audit the set.
const banned = problems.flatMap((p) =>
  p.approaches.filter((a) => a.doNotUseInInterview).map((a) => ({ p, a })),
);
console.log(
  `\nFlagged "do not use in interview": ${banned.length} approach(es) across ${new Set(banned.map((b) => b.p.slug)).size} problem(s).`,
);
for (const { p, a } of banned) console.log(`  - ${p.slug} [${a.id}]  ${a.doNotUseInInterview}`);

if (diagnostics.unparsedHeaders.length) {
  console.log(
    `\nWARNING — ${diagnostics.unparsedHeaders.length} line(s) looked like a problem header but did not parse:`,
  );
  for (const line of diagnostics.unparsedHeaders) console.log(`  ${line}`);
} else {
  console.log('\nNo header-shaped lines were skipped.');
}

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

if (failures.length) {
  console.error(`\nFAILED — ${failures.length} problem(s) with the parsed data:`);
  for (const message of failures) console.error(`  x ${message}`);
  process.exit(1);
}

console.log('\nOK — parsed data is consistent.');
