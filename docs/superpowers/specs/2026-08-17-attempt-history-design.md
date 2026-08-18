# Attempt history — design

**Date:** 2026-08-17
**Status:** approved

## Problem

Solutions in this repo are timeless. A header records what a problem is and which
approach solved it, but not *when* it was solved or how long it took, and re-solving
a problem the same way has nowhere to go — the parser merges blocks sharing a `[tag]`
into one approach whose language tabs are keyed by language, so a second Python
attempt collides with the first.

The author wants to log every attempt from now on, see repeat attempts of the same
method side by side to watch the approach evolve, and sort the problem list by
most recent activity.

## Header grammar

One optional date and one optional solve time, appended to the existing line:

```
# <num>. <Title> (<url>) - <Difficulty>[ - <YYYY-MM-DD>][ - <solve time>][ [tag]]
```

```python
# 36. Valid Sudoku (https://leetcode.com/problems/valid-sudoku/) - Medium - 2026-08-17 - 14m [three-passes]
# 347. Top K Frequent Elements (https://leetcode.com/...) - Medium - 2026-08-17 [max-heap]
# 217. Contains Duplicate (https://leetcode.com/...) - Easy [hash-set]
```

This extends the convention already used by 20+ daily-challenge entries
(`- Medium - 2026-01-05`), so those gain dates with no editing. The parser already
captures this tail into an unused `rest` capture group and discards everything but
the `[tag]`; this reads two more things out of it.

**Date** is strict `YYYY-MM-DD`. Stored and compared as a string — ISO dates sort
lexicographically, so no `Date` parsing anywhere.

**Solve time** is free text kept verbatim for display (`14m`, `9m 40sec`, `1h 5m`).
It is never parsed into a duration and never sorted on, because nothing needs that.
To avoid swallowing unrelated trailing text, a segment only counts as a solve time
if it matches `/\d+\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|seconds?)\b/i`.
This is what keeps the existing `- Medium - 2026-01-07 - Duplicate` header from
reporting a solve time of "Duplicate".

## Data model

`Implementation` gains two optional fields. An implementation already *is* one
source block, which is exactly one attempt, so no new nesting level is introduced.

```ts
export interface Implementation {
  lang: Lang;
  code: string;
  sourceFile: string;
  startLine: number;
  endLine: number;
  attemptedOn?: string;   // "2026-08-17"
  solveTime?: string;     // "14m", verbatim
}
```

`Problem` gains one derived field for sorting:

```ts
lastAttempt?: string;   // max attemptedOn across all implementations
```

## Rendering

**Tabs** (`LangTabs.astro`). The radio `id` changes from `${group}--${impl.lang}` to
`${group}--${i}`. That is the whole fix for the collision: today two Python blocks
under one tag produce duplicate DOM ids, so the second tab activates the first.

Tab labels show the date whenever present, regardless of whether it is needed to
disambiguate:

```
┌───────┬─────────────────────┬─────────────────────┐
│ Java  │ Python · 2026-08-17 │ Python · 2026-12-01 │
└───────┴─────────────────────┴─────────────────────┘
```

Each panel gets a meta line above the code reading `2026-08-17 · 14m`, omitting
whichever part is absent. Undated attempts render exactly as they do today.

**Tab cap** (`global.css`). The `:nth-of-type` rules go from 5 slots to 8. Attempts
accumulate over time and the current behavior on overflow is `slice(0, 5)` — silent
truncation, the wrong failure mode for an archive.

**Sorting** (`index.astro`). A "Recent" toggle joins the existing search / difficulty
/ category controls, reusing the `[data-row]` filter machinery already on the page.
Problems without a `lastAttempt` sort last, preserving their current relative order.

## Non-goals

- **A Timeline page.** A per-attempt feed duplicates what the problem page already
  shows as tabs, one click away. The date lives in the data model either way, so this
  stays a pure rendering addition if it is ever wanted.
- **Side-by-side diffing.** Two browser tabs do this.
- **Per-attempt prose.** Approach-level `explanation` / complexity / pitfalls keep
  their current first-non-empty-wins behavior. A repeat attempt contributes its date
  and solve time, not a second notes block; that would mean moving prose inside each
  tab panel, a layout change this does not need.
- **Backfilling undated problems.** They sort last and show no date. Fine.

## Verification

- `npm run verify` — parses the whole repo, must report 0 unparsed headers.
- `npm run build` — must build 155+ pages. The count is the regression test: a
  silent drop of dynamic routes is exactly the failure mode seen on the Astro 7
  branch, and page count catches it where exit code does not.
- Existing dated daily-challenge entries must surface dates without edits.
- `- Medium - 2026-01-07 - Duplicate` must yield a date and no solve time.

## Out of scope, noted

The working tree on `main` carries an uncommitted Astro 5→7 upgrade under which
`astro build` produces 4 pages instead of 155 — every `getStaticPaths` route silently
vanishes, exit code 0. This branch is cut from `HEAD` (Astro 5.18.2) and does not
touch it.
