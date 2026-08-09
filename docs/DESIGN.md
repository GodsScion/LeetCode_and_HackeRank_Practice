# System Design — Solutions Site

How this repo becomes <https://godsscion.github.io/LeetCode_and_HackeRank_Practice/>.

## The one constraint everything else follows from

**The solution files are the source of truth and are never modified by the build.**

`Leet Code/`, `Hacker Rank/`, and `scratch pad/` are read-only inputs. There is no
database, no CMS, no per-problem markdown to keep in sync, and no migration. You keep
appending to `python.py` exactly as you have for two years; the site re-derives itself
on every push.

This is the whole design. Everything below is a consequence.

## Why the obvious approach doesn't work

The original idea was to iframe the LeetCode problem on the left and show solutions on
the right. That is not possible:

```
leetcode.com     x-frame-options: SAMEORIGIN
hackerrank.com   x-frame-options: SAMEORIGIN
```

Both refuse to be framed. A proxy that re-serves their HTML would work technically but
is fragile and reproduces their copyrighted content, which is worse than the problem it
solves. So the left pane holds **your own** metadata and links out instead.

## Pipeline

```
Leet Code/python.py ┐
Leet Code/java.java │
Leet Code/*.js|ts|c ├──► parser.ts ──► SiteData ──► Astro ──► dist/ ──► GitHub Pages
Hacker Rank/*.py    │    (read-only)   (typed)      (+Shiki)
scratch pad/*       ┘
```

One parser, one typed object, one static build. No runtime, no server, no client-side
data fetching.

## Data model

The key modelling decision: **the approach is the primary entity, not the file.**

```
Problem                     ← one per LeetCode number
  ├─ metadata               ← number, title, difficulty, category, links
  ├─ summary?               ← your words; optional, renders only if present
  └─ Approach[]             ← a distinct algorithm (e.g. "two pointers")
       ├─ explanation, timeComplexity, spaceComplexity, pitfalls?
       └─ Implementation[]  ← the same approach in Python / Java / TS
```

A page reads: *Solution 1 — Sorted Key*, its complexity and explanation, then language
tabs. Language is an attribute of an approach, not a sibling of it.

That requires knowing which Python block and which Java block are the *same* approach.
Nothing in the repo said so, so approaches are grouped by an optional tag on the header
comment you already write:

```python
# 49. Group Anagrams (https://leetcode.com/problems/group-anagrams/) - Medium [sorted-key]
```
```java
// 49. Group Anagrams (https://leetcode.com/problems/group-anagrams/) - Medium [sorted-key]
```

Matching tags merge into one approach with two tabs. **Untagged blocks still work** —
each becomes its own numbered approach. The site is correct today with zero annotation
and gets better as tags are added. Tags live on the header line rather than in a sidecar
file specifically so they cannot drift out of sync with the code.

Complexity and prose need no new format at all: your existing docstring convention
already is the schema.

```python
class Solution:
    '''
    Time Complexity: O(n)      → badge
    Space Complexity: O(n)     → badge
    Used BFS to do level order traversal    → explanation prose
    '''
```

73% of blocks already carry complexity. The rest simply render without badges.

## Stack, and why each piece earns its place

| Piece | Why | Why not the alternative |
|---|---|---|
| **Astro** | Static HTML out, file-based routing, zero JS by default | Next.js ships a React runtime for a site that is 99% static text |
| **Shiki** | VS Code's own TextMate grammars — correct Java generics and Python type hints; runs at build time | Prism/highlight.js visibly mangle both, and highlight in the browser |
| **Tailwind v4** | Design system via CSS-first `@theme`, no config file, no bespoke CSS to maintain | Hand-written CSS is more code and drifts |
| **Vanilla JS filter** | 140 problems is ~15KB of JSON; filtering is ~30 lines | Pagefind adds a WASM index and a build step to search a list that fits in memory |
| **CSS-only tabs** | `:checked ~` sibling selectors; keyboard-accessible for free | A JS tab component is more code and a hydration boundary |

Four dependencies total. **shadcn/ui was considered and rejected**: it requires React +
Radix, a full client runtime, to provide components this site can express in CSS.

Non-goals, deliberately: no code execution, no light theme, no comments, no analytics,
no auth. Each would add moving parts to a reference site that needs none.

## Legal position

- **Your code and notes** — yours, MIT, already.
- **Problem statements** — LeetCode's copyrighted content, and reproducing them breaks
  their ToS. **The site does not reproduce them.** It stores only number, title,
  difficulty, and a link, which are facts and not protectable.
- **Category names** come from your own section banners, not LeetCode's taxonomy.
- The NeetCode link is derived from the LeetCode slug (`neetcode.io/solutions/<slug>`);
  it is an outbound link, nothing is embedded.

## Failure mode this design protects against

The previous attempt at this (`Syntax-Shenanigans`) was scaffolded and abandoned the
same day. The blocker was never the framework — it was that there was no path from a
5000-line append-only `python.py` to a set of pages, so every new solution meant hand-
writing HTML.

Here, adding a solution is appending to `python.py` exactly as before. The site updates
itself. CI runs `verify-parse` before every build, so a malformed header comment fails
the workflow loudly instead of silently dropping a solution from the site.
