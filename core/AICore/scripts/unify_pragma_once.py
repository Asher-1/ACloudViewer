#!/usr/bin/env python3
"""
Convert all `#ifndef` include guards in AICore headers to `#pragma once`.

Handles two cases:
  A) `#ifndef` only (no `#pragma once`) — add `#pragma once`, remove ifndef/define/endif.
  B) Both `#pragma once` and `#ifndef`  — keep `#pragma once`, remove ifndef/define/endif.
"""

import os
import re
import sys

AICORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_DIRS = [
    os.path.join(AICORE_ROOT, "src"),
    os.path.join(AICORE_ROOT, "include"),
    os.path.join(AICORE_ROOT, "tests"),
]


def find_guard_lines(lines):
    """Find the `#ifndef ... #define ...` guard pair and trailing `#endif`.

    Returns (guard_start, guard_close) 0-based line indices or None if not found.
    guard_start points to the first line after `#define`, guard_close to `#endif`.
    """
    guard_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#ifndef ") and not stripped.startswith("#ifndef __"):
            guard_start = i
            break
    if guard_start is None:
        return None

    # Expect `#define GUARD` on the next line (or next non-blank, non-comment)
    define_line = None
    for j in range(guard_start + 1, min(guard_start + 4, len(lines))):
        stripped = lines[j].strip()
        if stripped == "" or stripped.startswith("//") or stripped.startswith("/*"):
            continue
        if stripped.startswith("#define "):
            define_line = j
            break
    if define_line is None:
        return None

    # Find trailing `#endif` (last non-blank, non-closing-brace line in the file)
    guard_close = None
    for i in range(len(lines) - 1, max(define_line, len(lines) - 30) - 1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("#endif"):
            guard_close = i
            break
    if guard_close is None:
        return None

    return (guard_start, define_line, guard_close)


def has_pragma_once(lines):
    return any(line.strip() == "#pragma once" for line in lines)


def process_file(filepath, dry_run=False):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    guard = find_guard_lines(lines)
    if guard is None:
        return False

    guard_start, define_line, guard_close = guard
    has_pragma = has_pragma_once(lines)

    if has_pragma:
        # Case B: already has #pragma once, just strip the old guard artifacts
        # Mark lines for removal: #ifndef, #define, trailing #endif
        to_remove = {guard_start, define_line, guard_close}
    else:
        # Case A: no #pragma once yet; add it and strip guard artifacts
        to_remove = {guard_start, define_line, guard_close}

    if dry_run:
        rel = os.path.relpath(filepath, AICORE_ROOT)
        print(f"  {'BOTH' if has_pragma else 'IFNDEF'} {rel}: "
              f"remove lines {guard_start+1}, {define_line+1}, {guard_close+1} "
              f"{'' if has_pragma else '+ add #pragma once'}")
        return True

    new_lines = []
    pragma_added = False
    for i, line in enumerate(lines):
        if i in to_remove:
            continue
        if not has_pragma and not pragma_added and i > guard_start:
            # Insert #pragma once after the last removed guard line
            # Find a good insertion point: right before first content line after license
            if i > define_line:
                # Check if preceding kept line is blank or comment block end
                if i == define_line + 1:
                    inserted = False
                    # Look for first "content" line (non-blank, not copyright)
                    for k in range(define_line + 1, guard_close):
                        if k in to_remove:
                            continue
                        stripped = lines[k].strip()
                        if not inserted and (stripped == "" or
                                             stripped.startswith("//")):
                            continue
                        new_lines.append("#pragma once\n\n")
                        pragma_added = True
                        inserted = True
                        break
                new_lines.append(line)
                continue
        new_lines.append(line)

    # If pragma wasn't added yet, add it right before the first content line
    if not has_pragma and not pragma_added:
        # Find insertion point after the guard block
        insert_after = max(guard_start, define_line)
        # Insert right after the `#define` line
        result = []
        inserted = False
        for i, line in enumerate(lines):
            if i == define_line:
                result.append("#pragma once\n\n")
                inserted = True
            elif i not in to_remove:
                result.append(line)
        if not inserted:
            result = ["#pragma once\n\n"] + [
                l for i, l in enumerate(lines) if i not in to_remove
            ]
        new_lines = result
        pragma_added = True

    # Strip trailing whitespace from the end-of-file
    while new_lines and new_lines[-1].strip() == "":
        new_lines.pop()

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    return True


def main():
    dry_run = "--dry-run" in sys.argv

    extensions = (".hpp", ".h")
    files = []
    for d in TARGET_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _dirs, _files in os.walk(d):
            for f in _files:
                if f.endswith(extensions):
                    files.append(os.path.join(root, f))

    files.sort()
    total = 0
    for fp in files:
        if process_file(fp, dry_run=dry_run):
            total += 1

    if dry_run:
        print(f"\nWould modify {total} files.")
    else:
        print(f"\nModified {total} files.")


if __name__ == "__main__":
    main()