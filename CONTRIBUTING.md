# Contributing

Mostly a note to my future self.

The solution files are the source of truth. The site is generated from them by parsing the
header comment above each solution. Get the header right and everything else follows; get
it wrong and CI fails.

The reliable way to add a problem is to not type the header at all:

```bash
cd site && npm run new-problem -- <problem-url>
```

## The header contract

One line above each solution:

```python
# 217. Contains Duplicate (https://leetcode.com/problems/contains-duplicate/description/) - Easy
```

```java
// 217. Contains Duplicate (https://leetcode.com/problems/contains-duplicate/description/) - Easy
```

```python
#<< Balanced Brackets (https://www.hackerrank.com/challenges/balanced-brackets/problem) - Medium
#>>
```

Rules:

- `#` in Python, `//` in Java, JavaScript, TypeScript and C.
- LeetCode: `<number>. <Title> (<url>) - <Easy|Medium|Hard>`. The number and the difficulty
  are both required.
- HackerRank: opens with `#<<`, has no number, and **must** be closed with `#>>` on its own
  line. That closer is what bounds the block.
- The URL goes in parentheses, with nothing else inside them.
- A `####### SECTION NAME #######` banner sets the category for every solution below it,
  until the next banner. Put a new problem under the right banner, not at the end of the file.

## The `[tag]` convention

By default every block is its own approach. A matching `[tag]` on two or more blocks merges
them into a single approach on the site, with one tab per language.

This is the whole reason tags exist: so one approach solved in two languages shows up as one
approach with two tabs, not as two unrelated entries.

`Leet Code/python.py`:

```python
# 49. Group Anagrams (https://leetcode.com/problems/group-anagrams/description/) - Medium [sorted-key]
class Solution:
    '''
    Time Complexity: O(n·k log k)
    Space Complexity: O(n·k)
    Where n is the number of words and k is the longest word.
    '''
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ...
```

`Leet Code/java.java`:

```java
    // 49. Group Anagrams (https://leetcode.com/problems/group-anagrams/description/) - Medium [sorted-key]
    public List<List<String>> groupAnagrams(String[] strs) {
        ...
    }
```

Those two render as one approach, "Sorted Key", with a Python tab and a Java tab.

A different approach to the same problem gets a different tag (or no tag):

```python
# 49. Group Anagrams (https://leetcode.com/problems/group-anagrams/description/) - Medium [char-count]
```

Tag rules: lowercase letters, digits and single dashes (`two-pointers`, `dp-bottom-up`).
Name it after the idea, not the language. Untagged blocks still work — they just each become
their own numbered approach, so tags are an upgrade, never a prerequisite.

## Where complexity and notes go

In the docstring directly under `class Solution:`, as plain lines:

```python
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(1)
    Pitfalls: Using `>=` instead of `>` breaks the duplicate case.

    Anything else here is kept as prose and rendered above the code.
    '''
```

- `Time Complexity:`, `Space Complexity:` and `Pitfalls:` are pulled out into their own
  fields on the page. Everything else in the docstring renders as the explanation.
- These belong to the *approach*, not the language — so for a tagged approach implemented in
  several languages, write them once, on the Python side. `new-problem` leaves a block
  comment on the Java/JS/TS side too; fill that in only if there is no Python implementation.
- Leaving them out is fine. The fields just don't render.

## CI

Every push and pull request runs the parser over the whole repo before building:

```bash
cd site && npm run verify
```

If a header comment doesn't parse, the workflow fails. This is deliberate — the alternative
is a solution silently vanishing from the site because of a missing `- Medium`. Run it
locally before pushing.

Pull requests run verify and build but do not deploy. Only `main` deploys.
