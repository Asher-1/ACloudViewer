#!/usr/bin/env python3
"""
Convert all private relative #include directives in AICore source to full paths.

For src/ source files: resolves relative to core/AICore/src/
For tests/ source files: resolves relative to core/AICore/

The src/ directory will be added as a PRIVATE include dir in CMakeLists.txt;
the tests/ already have the parent as include dir.

Algorithm:
  1. Index every .hpp/.h under src/ and tests/ with their relative path from
     the appropriate base.
  2. For each source file, parse #include "…" directives.
  3. Skip: angled, aicore/ (PUBLIC API), ggml*, CVTools.h.
  4. For bare filenames, look up in the index (using source dir as tiebreaker).
  5. Replace with the full path.
"""

import os
import re
import sys
from collections import defaultdict

AICORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AICORE_SRC = os.path.join(AICORE_ROOT, "src")
AICORE_TESTS = os.path.join(AICORE_ROOT, "tests")


# ---------------------------------------------------------------------------
# Header index
# ---------------------------------------------------------------------------

def build_header_index():
    """Return {filename: [(full_rel_path, directory), …]}.

    For headers under src/  → full_rel_path is relative to AICORE_SRC
    For headers under tests/ → full_rel_path is relative to AICORE_ROOT
    """
    idx = defaultdict(list)

    def scan(base_dir, base_rel_prefix):
        if not os.path.isdir(base_dir):
            return
        for root, _dirs, files in os.walk(base_dir):
            for f in files:
                if not (f.endswith(".hpp") or f.endswith(".h")):
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, base_rel_prefix)
                idx[f].append((rel, os.path.dirname(rel)))

    scan(AICORE_SRC, AICORE_SRC)
    scan(AICORE_TESTS, AICORE_ROOT)
    return idx


# ---------------------------------------------------------------------------
# Include resolution
# ---------------------------------------------------------------------------

# For includes like "vulkan/xxx.hpp", "cuda/xxx.hpp", "lightglue/xxx.h"
# the task dir needed to form the full path from src/
SUBDIR_TASK_MAP = {
    "vulkan/":    "tasks/aliked",
    "cuda/":      "tasks/aliked",
    "lightglue/": "tasks/aliked/include",
}

# Headers that are external (not in AICore src/tests)
EXTERNAL_HEADERS = frozenset({
    "CVTools.h",
    "immintrin.h",
    "ggml.h", "ggml-backend.h", "ggml-cpu.h", "ggml-alloc.h",
    "gguf.h",
})


def _best_match(entries, source_rel, header_index):
    """Pick the best candidate from entries (filename → [(path, dir), …]).

    Preference order:
      1. Same directory as the including file.
      2. Under a subdirectory of the including file's directory.
      3. Under common/.
      4. First entry.
    """
    source_dir = os.path.dirname(source_rel)

    # Try exact dir match (guaranteed no ambiguity for local includes)
    for path, dir_part in entries:
        if dir_part == source_dir:
            return path

    # Try hierarchical: source_dir starts with dir_part or vice versa
    for path, dir_part in entries:
        if source_dir.startswith(dir_part + "/") or dir_part.startswith(source_dir + "/"):
            return path

    # Try matching task name: for test files in tests/XXX/..., prefer
    # tasks/XXX/ over other tasks (e.g. tests/depth/whitebox/ → tasks/depth/)
    src_segments = [s for s in source_dir.split("/") if s and s != "tests"]
    if src_segments:
        for path, dir_part in entries:
            if dir_part.startswith("tasks/"):
                task = dir_part[len("tasks/"):]
                if task in source_dir or task in src_segments:
                    return path

    # Prefer common/
    for path, dir_part in entries:
        if dir_part == "common":
            return path
        if dir_part.startswith("tests/common"):
            return path

    return entries[0][0]


def resolve_include(target, source_rel, header_index):
    """Resolve #include "target" from a file at *source_rel*.

    source_rel is the path of the including file relative to the
    appropriate base (AICORE_SRC for src/, AICORE_ROOT for tests/).
    Returns the resolved full path (relative to same base) or None.
    """

    # ---- 1. Subdir-prefixed (vulkan/, cuda/, lightglue/) ----------------
    for prefix, task_dir in SUBDIR_TASK_MAP.items():
        if target.startswith(prefix):
            # Expected path under src/
            full = f"{task_dir}/{target}"
            if os.path.isfile(os.path.join(AICORE_SRC, full)):
                return full
            break

    # ---- 2. Already has a directory component ---------------------------
    if "/" in target:
        bases = []
        if source_rel.startswith("tests/"):
            # Test file: headers may be under tests/ or tasks/
            bases.append((AICORE_ROOT, target))           # tests/common/test_macros.hpp
            bases.append((AICORE_ROOT, f"tests/{target}"))
            bases.append((AICORE_SRC, f"tasks/{target}"))
        # Always try direct from both bases
        for base in (AICORE_SRC, AICORE_ROOT):
            if os.path.isfile(os.path.join(base, target)):
                return target
        # Try combined from source file's directory
        combined = os.path.normpath(os.path.join(os.path.dirname(source_rel), target))
        for base in (AICORE_SRC, AICORE_ROOT):
            if os.path.isfile(os.path.join(base, combined)):
                return combined
        # Try the header index as fallback (for targets like "facedetect/backend.hpp")
        # where we need to map "backend.hpp" to "tasks/facedetect/backend.hpp"
        fname = os.path.basename(target)
        subdir = os.path.dirname(target)
        if fname in header_index:
            for path, dir_part in header_index[fname]:
                if subdir and (dir_part == subdir or dir_part.endswith("/" + subdir)):
                    return path
        return None

    # ---- 3. Bare filename -----------------------------------------------
    if target not in header_index:
        return None
    return _best_match(header_index[target], source_rel, header_index)


# ---------------------------------------------------------------------------
# Source file processing
# ---------------------------------------------------------------------------

RE_INCLUDE = re.compile(r'^(\s*#\s*include\s+)"([^"]+)"(\s*)$')

EXTERNAL_PREFIXES = ("aicore/",)


def _is_external(target):
    if target.startswith(EXTERNAL_PREFIXES):
        return True
    if target in EXTERNAL_HEADERS:
        return True
    return False


def process_file(filepath, header_index, dry_run=False):
    """Convert bare/relative includes in *filepath* to full paths.

    Returns True if the file was modified.
    """
    # Determine source-rel base
    if filepath.startswith(AICORE_SRC):
        source_rel = os.path.relpath(filepath, AICORE_SRC)
    elif filepath.startswith(AICORE_TESTS):
        source_rel = os.path.relpath(filepath, AICORE_ROOT)  # → tests/foo.cpp
    else:
        return False

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    changed = False
    new_lines = []

    for lineno, line in enumerate(lines, 1):
        m = RE_INCLUDE.match(line)
        if not m:
            new_lines.append(line)
            continue

        prefix = m.group(1)
        target = m.group(2)
        suffix = m.group(3)

        if _is_external(target):
            new_lines.append(line)
            continue

        resolved = resolve_include(target, source_rel, header_index)
        if resolved is None:
            print(f"  WARN: {source_rel}:{lineno}: unable to resolve '#include \"{target}\"'", file=sys.stderr)
            new_lines.append(line)
            continue

        if resolved == target:
            new_lines.append(line)
            continue

        new_include = f'{prefix}"{resolved}"{suffix}\n'
        if dry_run:
            print(f"  WOULD CHANGE {source_rel}:{lineno}: \"{target}\" -> \"{resolved}\"")
        new_lines.append(new_include)
        changed = True

    if changed and not dry_run:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    return changed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dry_run = "--dry-run" in sys.argv

    print("Building header index...")
    header_index = build_header_index()
    total_headers = sum(len(v) for v in header_index.values())
    print(f"  {total_headers} headers ({len(header_index)} unique names)")

    extensions = (".cpp", ".hpp", ".h")
    files = []
    for base in (AICORE_SRC, AICORE_TESTS):
        if not os.path.isdir(base):
            continue
        for root, _dirs, _files in os.walk(base):
            for f in _files:
                if f.endswith(extensions):
                    files.append(os.path.join(root, f))

    files.sort()
    print(f"Processing {len(files)} source files...")

    total = 0
    for fp in files:
        if process_file(fp, header_index, dry_run=dry_run):
            total += 1

    if dry_run:
        print(f"\nWould modify {total} files.")
    else:
        print(f"\nModified {total} files.")


if __name__ == "__main__":
    main()