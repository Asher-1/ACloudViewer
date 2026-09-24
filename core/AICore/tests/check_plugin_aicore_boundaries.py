#!/usr/bin/env python3
"""Check that AICore plugins consume the public contract only.

The check is intentionally conservative: it only applies the stronger model
ownership rules to plugin directories that include a public ``aicore/*``
header. This keeps unrelated Qt plugins out of the AICore contract while
catching a new AI plugin before private implementation details leak into it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
PUBLIC_INCLUDE = re.compile(r"#\s*include\s*[<\"]aicore/")
FORBIDDEN_INCLUDE = re.compile(
    r"#\s*include\s*[<\"](?:ggml|gguf|.*(?:core/AICore/src|AICore/src|src/tasks)/)")
FORBIDDEN_ENV = re.compile(r"\b(?:getenv|std::getenv|qEnvironmentVariable)\s*\(")
# A drive-letter path must be a real path prefix.  ``db://`` and prose such
# as ``error:\n`` otherwise look like ``b:/`` and ``r:\n`` to a broad regex.
ABSOLUTE_PATH = re.compile(
    r"(?:/(?:home|Users|tmp|mnt|opt|data)/|"
    r"(?<![A-Za-z])[A-Za-z]:[\\/](?![nrt])|\\\\[^\\]+\\[^\\]+)")
PRIVATE_MODEL_OWNER = re.compile(
    r"(?:aicore/asset_digests\.h|QNetworkAccessManager|QNetworkReply|"
    r"cloudViewer_downloads/releases/download|huggingface\.co/.*/resolve/)",
    re.IGNORECASE)


def source_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*")
                  if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES)


def check_plugin_root(root: Path, plugin: str | None = None,
                      strict: bool = False) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    plugin_roots = [root / "plugins" / "core" / "Standard",
                    root / "plugins" / "core" / "IO"]
    for parent in plugin_roots:
        if not parent.is_dir():
            continue
        for directory in sorted(path for path in parent.iterdir() if path.is_dir()):
            if plugin and directory.name != plugin:
                continue
            files = source_files(directory)
            if not files:
                continue
            text_by_file = {
                path: path.read_text(encoding="utf-8", errors="replace")
                for path in files
            }
            if not any(PUBLIC_INCLUDE.search(text) for text in text_by_file.values()):
                continue
            for path, text in text_by_file.items():
                relative = path.relative_to(root)
                # Test fixtures intentionally use environment variables to
                # select models; the production plugin sources do not get
                # that exception.
                is_test = "tests" in path.parts
                for line_no, line in enumerate(text.splitlines(), 1):
                    if FORBIDDEN_INCLUDE.search(line):
                        issues.append(f"{relative}:{line_no}: private AICore/ggml include")
                    if FORBIDDEN_ENV.search(line) and not is_test:
                        issues.append(f"{relative}:{line_no}: environment-controlled plugin behavior")
                    # Test assertions may mention the catalog URL or digest;
                    # boundary debt is about production ownership, so keep
                    # those fixtures out of the migration count.
                    if (not is_test and not line.lstrip().startswith("//") and
                            ABSOLUTE_PATH.search(line)):
                        warnings.append(f"{relative}:{line_no}: developer absolute path")
                    if (not is_test and not line.lstrip().startswith("//") and
                            PRIVATE_MODEL_OWNER.search(line)):
                        warnings.append(f"{relative}:{line_no}: plugin-owned model/cache/download symbol")
    if strict:
        issues.extend(warnings)
        warnings = []
    return issues, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[3])
    parser.add_argument("--plugin", help="check one plugin directory by name")
    parser.add_argument("--strict", action="store_true",
                        help="promote ownership/path warnings to failures")
    args = parser.parse_args()
    issues, warnings = check_plugin_root(args.root.resolve(), args.plugin,
                                         args.strict)
    if issues:
        print("AICore plugin boundary check failed:", file=sys.stderr)
        print("\n".join(f"- {issue}" for issue in issues), file=sys.stderr)
        return 1
    if warnings:
        print("AICore plugin boundary check: PASS with legacy warnings",
              file=sys.stderr)
        print("\n".join(f"- {warning}" for warning in warnings),
              file=sys.stderr)
    print("AICore plugin boundary check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
