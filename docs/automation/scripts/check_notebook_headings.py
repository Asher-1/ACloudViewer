"""Guard: notebooks must not contain standalone Markdown thematic breaks.

Why this is a hard error rather than a style preference
------------------------------------------------------
nbsphinx converts each Markdown cell to RST with pandoc and concatenates the
results. Pandoc turns a standalone rule line ("---", "***", "___") into an RST
*transition*, and docutils rejects a document or section that BEGINS with one:

    ERROR/3  Document or section may not begin with a transition.

These notebooks used rules as visual separators directly beneath headings, so
the transition always landed at a section start. Under `sphinx-build -W` that
ERROR aborts the entire documentation build.

The rule is checked at the Markdown level on purpose: it is the exact
construct that produces the illegal transition, it needs no pandoc, and it
runs in milliseconds instead of after a 30-minute Docker build.
"""

from __future__ import annotations

import glob
import json
import re
import sys

# A thematic break: the whole line is one of - * _ repeated at least 3 times.
RULE = re.compile(r"^[-*_]{3,}$")


def offending_rules(path: str) -> list[tuple[int, str]]:
    """Return (cell index, line) for every standalone rule in a notebook."""
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    out = []
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        for line in cell.get("source", []):
            text = line.rstrip("\n")
            if RULE.match(text.strip()):
                out.append((index, text.strip()))
    return out


def main(argv: list[str]) -> int:
    paths = argv[1:] or sorted(glob.glob("docs/jupyter/**/*.ipynb", recursive=True))
    bad = 0
    for path in paths:
        try:
            rules = offending_rules(path)
        except Exception as exc:                     # unreadable or invalid JSON
            print(f"FAIL {path}: {exc}")
            bad += 1
            continue
        if rules:
            bad += 1
            print(f"FAIL {path}")
            for index, text in rules:
                print(f"     cell {index}: standalone rule {text!r}")
            print("     -> pandoc emits an RST transition here; docutils rejects")
            print("        a section that begins with one, and -W makes it fatal.")
            print("        Delete the line, or use a heading instead.")
    print(f"checked {len(paths)} notebook(s), {bad} with standalone rules")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
