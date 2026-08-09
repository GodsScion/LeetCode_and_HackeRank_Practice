# LeetCode and HackerRank Practice

[![Deploy site](https://github.com/GodsScion/LeetCode_and_HackeRank_Practice/actions/workflows/deploy.yml/badge.svg)](https://github.com/GodsScion/LeetCode_and_HackeRank_Practice/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

My worked solutions to LeetCode and HackerRank problems, in Python, Java, JavaScript,
TypeScript and C — along with the notes, complexity analysis and pitfalls I collected
while solving them.

**Browse them at [godsscion.github.io/LeetCode_and_HackeRank_Practice](https://godsscion.github.io/LeetCode_and_HackeRank_Practice/)**

## How this works

The solution files are the source of truth. Nothing generates them and nothing edits them.

The site in [`site/`](site) is an [Astro](https://astro.build) app that reads those files
at build time, parses the header comment above each solution, and renders the result as a
browsable, searchable site. The parser is strictly read-only — it never writes to, moves,
reorders or reformats anything under `Leet Code/`, `Hacker Rank/` or `scratch pad/`.

So the workflow stays what it always was: open the language file, write the solution under
a header comment. The site follows.

CI re-runs the parser on every push and pull request. If a header comment doesn't parse,
the build fails rather than quietly dropping that solution from the site.

> **Problem statements are deliberately not reproduced here.** They are LeetCode's and
> HackerRank's copyrighted content. Every problem on the site links out to the original
> instead. What's mine — the solutions, the notes, the complexity analysis — is MIT licensed.

## The header comment format

One comment line above each solution is all the site needs.

| Where | Format |
| --- | --- |
| LeetCode, Python | `# 217. Contains Duplicate (https://leetcode.com/problems/contains-duplicate/description/) - Easy` |
| LeetCode, Java / JS / TS / C | `// 217. Contains Duplicate (https://leetcode.com/problems/contains-duplicate/description/) - Easy` |
| HackerRank | `#<< Balanced Brackets (https://www.hackerrank.com/challenges/balanced-brackets/problem) - Medium` … closed by `#>>` |

The parts are: number, title, URL in parentheses, and `- Easy` / `- Medium` / `- Hard`.
HackerRank challenges have no number, and their blocks are explicitly closed with `#>>`.

A section banner sets the category for everything below it, until the next banner:

```python
####### ARRAYS AND HASHING #######
```

An optional `[tag]` suffix groups solutions into one approach with per-language tabs:

```python
# 49. Group Anagrams (https://leetcode.com/problems/group-anagrams/description/) - Medium [sorted-key]
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contract, including where complexity
notes go.

## Adding a problem

Don't hand-type the header — let the helper build it:

```bash
cd site
npm install
npm run new-problem -- https://leetcode.com/problems/two-sum/
```

It derives what it can from the URL, asks for the rest, shows you exactly what it will
insert and where, and waits for confirmation. It only ever adds lines. Use `--dry-run` to
preview without writing, and `--help` for the flags.

Then write the solution in the stub it left you.

## Running the site locally

```bash
cd site
npm install
npm run dev      # http://localhost:4321/LeetCode_and_HackeRank_Practice/
```

Other scripts:

```bash
npm run verify   # check every solution still parses — same check CI runs
npm run build    # production build into site/dist/
npm run preview  # serve the production build
```

Requires Node 24 (the scripts use native TypeScript type stripping).

## Layout

| Path | What's in it |
| --- | --- |
| `Leet Code/` | LeetCode solutions, one file per language (`python.py`, `java.java`, `javascript.js`, `typesript.ts`, `c.c`) |
| `Leet Code/java/212/` | The one problem that needed multiple Java classes in its own files |
| `Hacker Rank/python.py` | HackerRank challenges, with the original stub code kept inline for context |
| `Interviews/` | Problems from real interview loops (not published to the site) |
| `Others/` | Older one-off scripts and experiments (not published to the site) |
| `scratch pad/` | Working files — kept because half-finished thinking is worth keeping |
| `site/` | The Astro site that renders the solutions, notes and scratch pad |
| `site/src/lib/parser.ts` | The parser: header comments in, structured data out |
| `site/scripts/` | `new-problem.ts` (add a stub) and `verify-parse.ts` (the CI check) |

The general lessons at the top of `Leet Code/python.py` — the numbered `NOTES` block — are
pulled onto the site too. That block is the most useful thing in this repo.

## License

[MIT](LICENSE) © Sai Vignesh Golla
