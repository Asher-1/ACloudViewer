#!/usr/bin/env python3
"""Audit the locally recorded COLMAP alignment matrix.

This is intentionally a source-of-truth check, not a claim generator: a
release gate fails when an in-scope capability is not implemented or when its
entrypoint/validation evidence is missing. Deferred capabilities are retained
for provenance but explicitly excluded from the active hardware matrix.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _pkg_config_exists(name: str) -> bool:
    if not _command_exists("pkg-config"):
        return False
    return subprocess.run(
        ["pkg-config", "--exists", name], check=False
    ).returncode == 0


def probe_prerequisites(repo_root: Path) -> dict[str, bool]:
    """Probe build inputs without downloading or mutating the worktree."""
    poselib = any(
        (repo_root / candidate).is_file()
        for candidate in (
            "3rdparty/PoseLib/PoseLib/solvers/relpose_6pt_onesided_focal.h",
            "build_app/_deps/poselib-src/include/PoseLib",
        )
    ) or (repo_root / "build_app/_deps/poselib-src/PoseLib").is_dir()
    caspar = (repo_root / "3rdparty/Symforce-Caspar/generated/f32").is_dir()
    loma_gguf = any(
        path.suffix.lower() == ".gguf"
        and "loma" in path.name.lower()
        for path in repo_root.rglob("*")
        if path.is_file()
    )
    freeimage_refs = False
    for root in (repo_root / "libs/Reconstruction", repo_root / "3rdparty"):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".h", ".cc", ".cpp", ".cmake", ".txt"}:
                try:
                    if "freeimage" in path.read_text(errors="ignore").lower():
                        freeimage_refs = True
                        break
                except OSError:
                    continue
        if freeimage_refs:
            break
    return {
        "nvcc": _command_exists("nvcc"),
        "hipcc": _command_exists("hipcc"),
        "openimageio_source_recipe": (
            "ExternalProject_Add(ext_openimageio"
            in (repo_root / "3rdparty/openimageio/openimageio.cmake").read_text(
                errors="ignore"
            )
        ),
        "openimageio_opencv_disabled": (
            "-DUSE_OPENCV=OFF"
            in (repo_root / "3rdparty/openimageio/openimageio.cmake").read_text(
                errors="ignore"
            )
        ),
        "poselib_source": poselib,
        "caspar_generated_source": caspar,
        "loma_gguf": loma_gguf,
        "no_stale_freeimage_references": not freeimage_refs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parents[1]
        / "libs/Reconstruction/colmap_alignment_manifest.json",
    )
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--probe", action="store_true", help="print local prerequisites")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.probe or args.release:
        print("local prerequisites:")
        deferred_names = set()
        for item in manifest["capabilities"]:
            if item.get("status") == "deferred" and item.get("name") == "hip_patchmatch":
                deferred_names.add("hipcc")
        for name, available in probe_prerequisites(args.manifest.parents[1].parent).items():
            if name in deferred_names:
                print(f"  {'deferred':7s} {name} (outside active hardware matrix)")
            else:
                print(f"  {'ready' if available else 'missing':7s} {name}")
    failures = []
    for item in manifest["capabilities"]:
        status = item.get("status")
        if args.release and status not in {"implemented", "deferred"}:
            failures.append(f"{item['name']}: status={status}")
        if status == "implemented":
            for key in ("entrypoint", "validation"):
                if not item.get(key):
                    failures.append(f"{item['name']}: missing {key}")
    print(f"COLMAP revision: {manifest['upstream_revision_checked']}")
    for item in manifest["capabilities"]:
        print(f"{item['status']:11s} {item['name']}")
        if (args.probe or args.release) and item.get("status") != "implemented":
            print(f"  reason: {item.get('reason', 'not recorded')}")
            print(f"  resolve: {item.get('resolution', 'not recorded')}")
    if failures:
        print("alignment audit failed:")
        print("\n".join(f"  {failure}" for failure in failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
