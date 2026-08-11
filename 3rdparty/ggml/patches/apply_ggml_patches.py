#!/usr/bin/env python3
"""Apply ordered unified-diff patches to a fetched ggml source tree.

All ggml source modifications in ACloudViewer are checked in as *.patch
files under this directory and listed in manifest.yaml. The former in-place
Python mutators (apply_cpu_all_variants_compiler_checks.py,
apply_metal_conv_transpose_opt.py, apply_metal_fa_large_seq.py) were converted
into unified-diff patches so the fetched tree is byte-reproducible on every
platform. New work must add *.patch files to manifest.yaml.

Idempotency:
  - GNU `patch -p1 -N` skips hunks already applied (safe in ExternalProject trees
    that live inside the ACloudViewer git work tree — `git apply` can false-positive).
  - Non-zero exit -> fail (conflict / ggml version drift)

Usage: python3 apply_ggml_patches.py <ggml_source_dir> [manifest.yaml]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def _parse_manifest_fallback(text: str) -> list[str]:
    """Minimal parser for manifest.yaml when PyYAML is unavailable.

    Handles the simple format used by this project:
        patches:
          - file: path/to/patch.patch
            note: description
          - file: another/patch.patch
    """
    result: list[str] = []
    in_patches = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if stripped == "patches:":
            in_patches = True
            continue
        if in_patches and stripped.startswith("- file:"):
            path = stripped[len("- file:"):].strip().strip('"').strip("'")
            if path:
                result.append(path)
    return result


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def apply_one(src_dir: Path, patch_path: Path) -> bool:
    if not patch_path.is_file():
        print(f"[ggml-patch] missing patch file: {patch_path}")
        return False

    dry = _run(["patch", "-p1", "-N", "--dry-run", "-i", str(patch_path)], cwd=src_dir)
    dry_out = dry.stdout or ""
    if dry.returncode != 0 and not _patch_already_applied(dry_out):
        print(f"[ggml-patch] ERROR: cannot apply {patch_path.name}\n{dry_out}")
        return False

    if _patch_already_applied(dry_out):
        print(f"[ggml-patch] already applied: {patch_path.name}")
        return True

    applied = _run(["patch", "-p1", "-N", "-i", str(patch_path)], cwd=src_dir)
    out = applied.stdout or ""
    if applied.returncode != 0:
        print(f"[ggml-patch] ERROR applying {patch_path.name}\n{out}")
        return False

    print(f"[ggml-patch] applied: {patch_path.name}")
    return True


def _patch_already_applied(output: str) -> bool:
    markers = (
        "Skipping patch",
        "previously applied",
        "already exists",
        "hunk ignored",
        "Reversed (or previously applied) patch detected",
    )
    return any(marker in output for marker in markers)


def load_manifest(manifest_path: Path) -> list[Path]:
    if not manifest_path.is_file():
        return []

    manifest_text = manifest_path.read_text(encoding="utf-8")
    patches_dir = manifest_path.parent

    if yaml is not None:
        data = yaml.safe_load(manifest_text) or {}
        files = [e["file"] for e in data.get("patches", []) if isinstance(e, dict) and "file" in e]
        files += [e for e in data.get("patches", []) if isinstance(e, str)]
    else:
        print("[ggml-patch] PyYAML not available; using fallback parser")
        files = _parse_manifest_fallback(manifest_text)

    out: list[Path] = [patches_dir / f for f in files]
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: apply_ggml_patches.py <ggml_source_dir> [manifest.yaml]", file=sys.stderr)
        return 2

    src_dir = Path(sys.argv[1]).resolve()
    if not src_dir.is_dir():
        print(f"[ggml-patch] source dir not found: {src_dir}", file=sys.stderr)
        return 1

    manifest = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else Path(__file__).resolve().parent / "manifest.yaml"
    patch_files = load_manifest(manifest)
    if not patch_files:
        print("[ggml-patch] no manifest patches (ok)")
        return 0

    ok = True
    for patch_path in patch_files:
        if not apply_one(src_dir, patch_path):
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
