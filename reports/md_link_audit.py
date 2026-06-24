"""Audit + auto-fix `[text](#anchor)` references in a markdown file against actual
GitHub-style heading slugs.

GitHub slug rules (best-effort, sufficient for this repo):
  - Lowercase
  - Drop backticks/punctuation: . , ( ) [ ] : ; ? ! " ' & / \\ + = * ~ < >
  - Keep: word chars (Unicode letters incl. CJK), digits, '-', '_'
  - Spaces → '-'
  - Strip leading/trailing '-'
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Tuple, List


def slugify(heading: str) -> str:
    """Replicate github-slugger behaviour for our purposes.

    1. strip leading hashes / whitespace
    2. lowercase
    3. replace internal whitespace with '-'
    4. drop every char whose Unicode category is not L*/N* and that is not '_' or '-'
    """
    h = heading.strip().lstrip("#").strip()
    h = h.replace("`", "")
    h = h.lower()
    h = re.sub(r"\s+", "-", h)
    out_chars = []
    for ch in h:
        if ch in ("-", "_"):
            out_chars.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            out_chars.append(ch)
        # else: drop
    return "".join(out_chars).strip("-")


def collect_headings(text: str) -> List[Tuple[str, str]]:
    """Return list of (heading_text_with_hashes_stripped, slug)."""
    out = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if not m:
            continue
        title = m.group(2).rstrip()
        out.append((title, slugify(title)))
    return out


def collect_refs(text: str):
    """Yield (full_match, link_text, anchor, position)."""
    for m in re.finditer(r"\[([^\]]+)\]\(#([^\)]+)\)", text):
        yield m.group(0), m.group(1), m.group(2), m.start()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: md_link_audit.py <file> [--fix]")
        return 2
    fp = Path(sys.argv[1])
    fix = "--fix" in sys.argv
    text = fp.read_text(encoding="utf-8")
    headings = collect_headings(text)
    slug_set = {s for _, s in headings}

    refs = list(collect_refs(text))
    broken: list[tuple[str, str, str]] = []
    for full, label, anchor, _pos in refs:
        if anchor not in slug_set:
            broken.append((full, label, anchor))

    print(f"file: {fp}")
    print(f"headings: {len(headings)}  unique slugs: {len(slug_set)}")
    print(f"refs:     {len(refs)}      broken: {len(broken)}")
    if not broken:
        return 0

    # Build candidate map: try to match by §N. prefix or by token containment
    section_num_re = re.compile(r"^(\d+)[-．\.]")
    slug_by_num: dict[str, str] = {}
    for _t, s in headings:
        m = section_num_re.match(s)
        if m:
            slug_by_num.setdefault(m.group(1), s)

    fixes: list[tuple[str, str]] = []
    print("\n=== Broken refs (top 60) ===")
    for full, label, anchor in broken[:60]:
        # extract leading section number from label like "§53" or "§ 5.3"
        m = re.search(r"§\s*(\d+)", label)
        candidate = None
        if m and m.group(1) in slug_by_num:
            candidate = slug_by_num[m.group(1)]
        elif (m2 := re.match(r"^(\d+)", anchor)):
            if m2.group(1) in slug_by_num:
                candidate = slug_by_num[m2.group(1)]
        if candidate and candidate != anchor:
            new_full = full.replace(f"](#{anchor})", f"](#{candidate})")
            fixes.append((full, new_full))
            print(f"  FIX  {label!r:40s} #{anchor}\n        -> #{candidate}")
        else:
            print(f"  KEEP {label!r:40s} #{anchor}  (no obvious target)")

    if fix and fixes:
        new_text = text
        for old, new in fixes:
            new_text = new_text.replace(old, new)
        fp.write_text(new_text, encoding="utf-8")
        print(f"\nApplied {len(fixes)} fixes -> {fp}")
    elif fix:
        print("\nNo automatic fixes available.")
    else:
        print(f"\nDry run; would apply {len(fixes)} fixes. Re-run with --fix.")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
