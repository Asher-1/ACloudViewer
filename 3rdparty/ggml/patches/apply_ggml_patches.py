#!/usr/bin/env python3
"""Apply ordered unified-diff patches to a fetched ggml source tree.

All ggml source modifications in ACloudViewer are checked in as *.patch
files under this directory and listed in manifest.yaml. The former in-place
Python mutators (apply_cpu_all_variants_compiler_checks.py,
apply_metal_conv_transpose_opt.py, apply_metal_fa_large_seq.py) were converted
into unified-diff patches so the fetched tree is byte-reproducible on every
platform. New work must add *.patch files to manifest.yaml.

Idempotency:
  - The complete ordered chain is applied to a temporary source copy first.
  - If forward replay fails, a complete reverse replay identifies a source
    where the whole chain is already applied.
  - For sources nested in ACloudViewer's work tree, `git apply --directory`
    explicitly anchors every patch path at the fetched ggml source directory.
  - Any other result fails on conflicts, partial application, or version drift.

Usage: python3 apply_ggml_patches.py <ggml_source_dir> [manifest.yaml]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
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


def _run(
    cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _git_command(
    src_dir: Path, patch_path: Path, reverse: bool = False
) -> subprocess.CompletedProcess:
    git = os.environ.get("ACV_GIT_EXECUTABLE") or shutil.which("git")
    if not git:
        return subprocess.CompletedProcess([], 127, "git executable not found")

    git_env = os.environ.copy()
    command = [git, "apply", "--whitespace=nowarn"]
    cwd = src_dir
    repository = _run(
        [git, "-C", str(src_dir), "rev-parse", "--show-toplevel"], env=git_env
    )
    if repository.returncode == 0:
        repository_root = Path(repository.stdout.strip()).resolve()
        try:
            source_relative = src_dir.resolve().relative_to(repository_root)
        except ValueError:
            pass
        else:
            cwd = repository_root
            # When src_dir IS the repository root, no directory adjustment is
            # needed — patch paths are already relative to the root.  Adding
            # --directory=. produces paths like ./CMakeLists.txt which git
            # apply rejects as invalid.
            if source_relative != Path("."):
                command.append(
                    f"--directory={source_relative.as_posix()}"
                )
    if reverse:
        command.append("--reverse")
    command.append(str(patch_path))
    return _run(command, cwd=cwd, env=git_env)


def _try_sequence(src_dir: Path, patch_paths: list[Path], reverse: bool) -> tuple[bool, str]:
    """Apply a complete chain to a temporary source copy.

    Later patches may edit files introduced by earlier patches, so checking
    patches independently is not sufficient for either direction.
    """
    with tempfile.TemporaryDirectory(prefix="acloudviewer-ggml-patch-") as temp:
        scratch = Path(temp) / "src"
        shutil.copytree(src_dir, scratch)
        ordered = list(reversed(patch_paths)) if reverse else patch_paths
        for patch_path in ordered:
            result = _git_command(scratch, patch_path, reverse=reverse)
            if result.returncode != 0:
                direction = "reverse" if reverse else "forward"
                return (
                    False,
                    f"{direction} failed at {patch_path.name}:\n{result.stdout or ''}",
                )
    return True, ""


def _try_recover_partial(
    src_dir: Path, patch_paths: list[Path]
) -> tuple[bool, str]:
    """Recover a partially-patched source.

    A previous failed patch step can leave the source with some patches
    applied and others not (e.g. the chain grew after an interrupted build).
    Neither the full forward nor the full reverse chain then applies.  This
    un-applies whatever is applied (walking backwards from the tail), verifies
    the full chain re-applies cleanly on a scratch copy, and only then mutates
    the real source.  A genuinely drifted tree fails the scratch verification
    and is reported as unrecoverable.
    """
    with tempfile.TemporaryDirectory(prefix="acloudviewer-ggml-patch-") as temp:
        scratch = Path(temp) / "src"
        shutil.copytree(src_dir, scratch)

        # Un-apply applied patches, walking backwards from the tail.  Patches
        # that are not applied fail to reverse and are skipped.
        for patch_path in reversed(patch_paths):
            _git_command(scratch, patch_path, reverse=True)

        # The scratch must now accept the full forward chain.
        for patch_path in patch_paths:
            result = _git_command(scratch, patch_path)
            if result.returncode != 0:
                return False, (
                    f"recovery impossible at {patch_path.name}:\n{result.stdout or ''}"
                )

    # Recovery verified on scratch: mirror it on the real source.
    print("[ggml-patch] recovering partially-patched source")
    for patch_path in reversed(patch_paths):
        result = _git_command(src_dir, patch_path, reverse=True)
        if result.returncode == 0:
            print(f"[ggml-patch] un-applied: {patch_path.name}")

    for patch_path in patch_paths:
        result = _git_command(src_dir, patch_path)
        if result.returncode != 0:
            print(
                f"[ggml-patch] ERROR applying {patch_path.name}\n{result.stdout or ''}"
            )
            return False, result.stdout or ""
        print(f"[ggml-patch] applied: {patch_path.name}")
    return True, ""


def apply_sequence(src_dir: Path, patch_paths: list[Path]) -> bool:
    for patch_path in patch_paths:
        if not patch_path.is_file():
            print(f"[ggml-patch] missing patch file: {patch_path}")
            return False

    forward_ok, forward_error = _try_sequence(src_dir, patch_paths, reverse=False)
    if forward_ok:
        for patch_path in patch_paths:
            result = _git_command(src_dir, patch_path)
            if result.returncode != 0:
                print(
                    f"[ggml-patch] ERROR applying {patch_path.name}\n"
                    f"{result.stdout or ''}"
                )
                return False
            print(f"[ggml-patch] applied: {patch_path.name}")
        verified, verification_error = _try_sequence(
            src_dir, patch_paths, reverse=True
        )
        if not verified:
            print(
                "[ggml-patch] ERROR: git reported success but the complete "
                "patch chain is not present\n"
                f"{verification_error}"
            )
            return False
        return True

    reverse_ok, reverse_error = _try_sequence(src_dir, patch_paths, reverse=True)
    if reverse_ok:
        for patch_path in patch_paths:
            print(f"[ggml-patch] already applied: {patch_path.name}")
        return True

    # Partially-patched source (e.g. interrupted previous build): un-apply
    # what is applied and re-apply the whole chain.
    recovered, recovery_error = _try_recover_partial(src_dir, patch_paths)
    if recovered:
        return True

    print(
        "[ggml-patch] ERROR: manifest cannot be applied or reversed as a "
        "complete chain; source is partially patched or has drifted\n"
        f"{forward_error}\n{reverse_error}\n{recovery_error}"
    )
    return False


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

    return 0 if apply_sequence(src_dir, patch_files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
