#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Keep every *functional* copy of the app version derived from version.txt.

WHY THIS EXISTS
---------------
``libs/cloudViewer/version.txt`` is the only place a human edits a version.
Every other occurrence is a copy, and an unguarded copy drifts silently. This
script makes the copies behave like derived data: one source, mechanical
propagation, and a ``--check`` mode that fails CI the moment they disagree.

WHAT IS DELIBERATELY *NOT* TOUCHED
----------------------------------
"Derive everything" is the wrong rule, and following it blindly would break
the build. Three kinds of literal look like version numbers but must stay:

1. **Build-time sentinels.** ``scripts/QtInstallerFramework/**`` carries the
   literal ``3.9.0`` which ``cmake/PostInstall.cmake`` rewrites *in the copied
   build tree* via
   ``string(REPLACE "3.9.0" "${CLOUDVIEWER_VERSION}" ...)``. If the sentinel
   tracked version.txt, that REPLACE would become a no-op (search string ==
   replacement) and the installer would ship whatever number happened to sit
   in the template. The sentinel must remain a fixed, unique token.

2. **Unrelated third-party versions.** ``util/ci_utils.sh`` pins
   ``MATPLOT_LIB_TARGET_VERSION="3.9.4"`` — that is *matplotlib's* version. It
   coincidentally shares the major.minor with ours; changing it would break the
   macOS torch-ops install.

3. **Historical records.** CHANGELOG entries, release posts and migration notes
   describe what happened. Rewriting them would falsify history.

Usage:
    python3 util/sync_version.py            # apply derived values
    python3 util/sync_version.py --check    # verify only; exit 1 on drift
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "libs" / "cloudViewer" / "version.txt"

# ---------------------------------------------------------------------------
# Managed targets: files whose version literal MUST equal version.txt.
#
# Each entry is (relative path, compiled pattern, group index holding the
# version digits). Everything outside that group is preserved verbatim, so a
# rewrite touches only the number and leaves formatting intact.
# ---------------------------------------------------------------------------
MANAGED: list[tuple[str, re.Pattern[str], int]] = [
    (
        # Read by npm/bin/acloudviewer.js (channel routing) and by
        # npm-publish.yml (which version to publish). Drifting here ships the
        # wrong package or routes to a release that does not exist.
        "npm/package.json",
        re.compile(r'("version"\s*:\s*")(\d+\.\d+\.\d+)(")'),
        2,
    ),
]

# ---------------------------------------------------------------------------
# Frozen literal sets: reported for evidence, never modified.
# ---------------------------------------------------------------------------
FROZEN = [
    (
        "scripts/QtInstallerFramework",
        "3.9.0 sentinel replaced at build time by cmake/PostInstall.cmake; "
        "must not track version.txt",
    ),
    (
        "util/ci_utils.sh",
        "MATPLOT_LIB_TARGET_VERSION is matplotlib's version, not ours",
    ),
    (
        "CHANGELOG.md",
        "historical release record",
    ),
    (
        "docs/downloads_data.json",
        "generated from the GitHub Releases API at build time",
    ),
    (
        "scripts/platforms/mac/bundle/test_lib_bundle_wheel.py",
        "wheel filenames in test fixtures are arbitrary labels; the test "
        "asserts deployment targets, not the version string",
    ),
]

# Files that must RETAIN a placeholder so the value stays derived at render
# time. If a literal reappears here, a local build has written over the
# placeholder and the value will silently stop tracking version.txt.
PLACEHOLDER_GUARDS = [
    ("docs/index.html", "@VERSION@",
     "version badge; substituted by the deploy step"),
    ("docs/source/getting_started/installation.in.rst", "@cv_version@",
     "install commands; substituted by make_docs"),
    ("docs/source/getting_started/builddocs.in.rst", "@cv_version@",
     "build commands; substituted by make_docs"),
]


def read_version() -> str:
    """Return MAJOR.MINOR.PATCH from version.txt, or exit with a clear error."""
    if not VERSION_FILE.is_file():
        sys.exit(f"error: {VERSION_FILE} not found")

    parts: dict[str, str] = {}
    for line in VERSION_FILE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"CLOUDVIEWER_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)\s*$",
                         line.strip())
        if match:
            parts[match.group(1).lower()] = match.group(2)

    missing = {"major", "minor", "patch"} - parts.keys()
    if missing:
        sys.exit(
            f"error: {VERSION_FILE} is missing {', '.join(sorted(missing))}; "
            f"expected three lines CLOUDVIEWER_VERSION_MAJOR/MINOR/PATCH")
    return f"{parts['major']}.{parts['minor']}.{parts['patch']}"


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def apply_target(rel_path: str, pattern: re.Pattern[str], group: int,
                 version: str, check_only: bool) -> list[str]:
    """Return drift descriptions for one managed target, applying the fix."""
    path = REPO_ROOT / rel_path
    if not path.is_file():
        return [f"{rel_path}: file is missing"]

    original = path.read_text(encoding="utf-8")
    drift: list[str] = []

    def repl(match: re.Match[str]) -> str:
        found = match.group(group)
        if found == version:
            return match.group(0)
        drift.append(f"{rel_path}:{line_of(original, match.start())}: "
                     f"has {found}, version.txt says {version}")
        if check_only:
            return match.group(0)
        start, end = match.span(group)
        whole_start = match.start()
        return (match.group(0)[:start - whole_start] + version +
                match.group(0)[end - whole_start:])

    updated = pattern.sub(repl, original)

    if not check_only and updated != original:
        path.write_text(updated, encoding="utf-8")

    return drift


def report_frozen() -> None:
    print("Frozen literals (intentionally NOT derived):")
    for rel_path, reason in FROZEN:
        print(f"  · {rel_path} — {reason}")


def check_placeholders() -> list[str]:
    """Return problems for files that lost their version placeholder.

    A build step that rewrites the source tree turns a placeholder into a
    literal. The number then looks right once and is wrong forever after,
    with nothing failing. Guard the placeholder itself.
    """
    problems: list[str] = []
    for rel_path, placeholder, purpose in PLACEHOLDER_GUARDS:
        path = REPO_ROOT / rel_path
        if not path.is_file():
            problems.append(f"{rel_path}: file is missing")
            continue
        text = path.read_text(encoding="utf-8")
        if placeholder in text:
            print(f"  ✅ {rel_path} keeps {placeholder} ({purpose})")
        else:
            problems.append(
                f"{rel_path}: lost its {placeholder} placeholder ({purpose}). "
                f"A build wrote a literal over it, so the value no longer "
                f"tracks version.txt.")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description=
        "Derive functional version copies from libs/cloudViewer/version.txt")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify only; exit 1 if any managed copy disagrees with version.txt"
    )
    args = parser.parse_args()

    version = read_version()
    print(f"source of truth: libs/cloudViewer/version.txt -> {version}")
    print()

    all_drift: list[str] = []
    for rel_path, pattern, group in MANAGED:
        drift = apply_target(rel_path, pattern, group, version, args.check)
        all_drift.extend(drift)
        if not drift:
            print(f"  ✅ {rel_path}")

    print()
    print("Placeholder guards:")
    placeholder_problems = check_placeholders()
    all_drift.extend(placeholder_problems)

    if all_drift:
        print()
        for item in all_drift:
            print(f"  ❌ {item}", file=sys.stderr)
        print()
        if args.check:
            print(
                "Version drift detected. Run 'python3 util/sync_version.py' "
                "to derive these from version.txt.",
                file=sys.stderr,
            )
            report_frozen()
            return 1
        # In apply mode a missing placeholder is not auto-fixable: restoring a
        # placeholder requires knowing what it should substitute, which is a
        # human decision about the surrounding text.
        if placeholder_problems:
            report_frozen()
            return 1
    elif not args.check:
        print()
        print("All managed copies match version.txt.")

    print()
    report_frozen()
    return 0


if __name__ == "__main__":
    sys.exit(main())
