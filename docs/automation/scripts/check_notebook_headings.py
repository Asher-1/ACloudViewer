"""Guard: no notebook markdown may form an RST overline title.

nbsphinx converts each markdown cell to RST and concatenates the results, so a
cell ending in "---" whose successor cell starts with prose is read as a
section-title overline. That conflicts with the heading levels already
established in the notebook and makes `sphinx-build -W` fail the whole docs
build with "Title level inconsistent".

The check is deliberately made against docutils, the parser that actually
decides, rather than against a regex over the source: a regex cannot tell a
thematic break from an overline once blank lines and cell seams are involved.
"""

from __future__ import annotations

import glob
import json
import sys

from docutils.core import publish_doctree


def notebook_markdown(path: str) -> str:
    """Return the markdown cells of a notebook joined the way nbsphinx does."""
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    parts = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "markdown"
    ]
    return "\n".join(parts)


def blank_fenced_blocks(markdown: str) -> str:
    """Blank out ``` fences before handing the text to docutils.

    nbsphinx runs the markdown through pandoc, which turns a fenced block into
    a literal block; feeding raw fences to docutils instead makes the closing
    ``` look like an RST overline and reports a phantom title error. Removing
    the fence contents keeps the parser's attention on real prose.
    """
    out = []
    in_fence = False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else line)
    return "\n".join(out)


def overline_errors(markdown: str) -> list[str]:
    """Return docutils diagnostics that indicate a broken section title."""
    tree = publish_doctree(blank_fenced_blocks(markdown),
                           settings_overrides={"report_level": 2})
    out = []
    for node in tree.traverse():
        if node.tagname != "system_message":
            continue
        text = node.astext()
        if "section title" in text.lower() or "title level" in text.lower():
            out.append(text.strip().splitlines()[0])
    return out


def main(argv: list[str]) -> int:
    paths = argv[1:] or sorted(glob.glob("docs/jupyter/**/*.ipynb", recursive=True))
    bad = 0
    for path in paths:
        try:
            errors = overline_errors(notebook_markdown(path))
        except Exception as exc:                      # unreadable/broken JSON
            print(f"FAIL {path}: {exc}")
            bad += 1
            continue
        if errors:
            bad += 1
            print(f"FAIL {path}")
            for line in errors:
                print(f"     {line}")
    print(f"checked {len(paths)} notebook(s), {bad} with broken section titles")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
