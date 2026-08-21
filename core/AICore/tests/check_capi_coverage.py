#!/usr/bin/env python3
"""AICore C-API coverage audit.

Counts how many public aicore_*_capi.h entry points are referenced by the
test suite and the in-tree consumers (plugins/libs/tools). This is the
"interface coverage" counterpart to gcov line coverage: an API that is never
called by any test or consumer is dead surface that can rot silently.

Usage:
    python3 core/AICore/tests/check_capi_coverage.py

Exit codes:
    0  interface coverage >= COVERAGE_TARGET
    1  coverage below target (print the missing symbols)
"""

import glob
import os
import re
import sys

# Root of the repository (script lives in core/AICore/tests/).
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
COVERAGE_TARGET = 95

# Directories whose call sites count as "covered" (tests first).
CONSUMER_DIRS = [
    "core/AICore/tests",
    "core/AICore/tools",
    "plugins/core/Standard",
    "libs",
]


def api_functions():
    funcs = set()
    for h in glob.glob(f"{REPO}/core/AICore/include/aicore/*_capi.h"):
        with open(h, encoding="utf-8") as f:
            txt = f.read()
        for m in re.finditer(r"\b(aicore_[a-z0-9_]+)\s*\(", txt):
            funcs.add(m.group(1))
    return funcs


def referenced_functions():
    referenced = set()
    for base in CONSUMER_DIRS:
        for t in glob.glob(f"{REPO}/{base}/**/*.cpp", recursive=True):
            with open(t, encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            for m in re.finditer(r"\b(aicore_[a-z0-9_]+)\s*\(", txt):
                referenced.add(m.group(1))
    return referenced


def main():
    api = api_functions()
    ref = referenced_functions()
    covered = api & ref
    missing = sorted(api - ref)
    pct = 100 * len(covered) // len(api)
    print(f"AICore C-API coverage: {len(covered)}/{len(api)} ({pct}%)")
    if missing:
        print(f"Never referenced by tests or consumers ({len(missing)}):")
        for name in missing:
            print(f"  {name}")
    ok = pct >= COVERAGE_TARGET
    print(f"Target: {COVERAGE_TARGET}% -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
