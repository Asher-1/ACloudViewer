#!/usr/bin/env python3
"""Build the upstream VLX-Seek-GGML reference CLI for the vlx_seek probe.

The vlx_seek task's parity reference is the upstream Asher-1/VLX-Seek-GGML
C++ port at a pinned revision: ``test_vlx_probe --upstream-dir DIR`` runs
``vlx-cli run`` from DIR on the same models, image, task, text and device and
requires identical greedy tokens / matching boxes. That CLI must come from
the *pinned upstream revision* without AICore's task tree, so this script:

1. downloads (or verifies ``--archive`` of) the pinned upstream repository
   archive plus its pinned llama.cpp submodule archive;
2. extracts both pristine into ``<out>/src``;
3. configures the backend set of the production build (CUDA / Vulkan read
   from the ext_llama CMakeCache), CPU native;
4. builds ``vlx-cli`` and prints the binary directory.

Pass the printed directory to validate_all.py as
``--reference vlx_upstream=<bin dir>``. Nothing generated here is checked in.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OWNER = REPO / "3rdparty" / "llama" / "llama_cpp.cmake"

# Pinned upstream integration revision (the C++/GGML port commit) and its
# llama.cpp submodule commit (.gitmodules gitlink).
UPSTREAM_REPO = "https://github.com/Asher-1/VLX-Seek-GGML"
UPSTREAM_REVISION = "e63d16d827fd91e087e034ff034cc035913f5c79"
UPSTREAM_LLAMA_SUBMODULE = "e467e5b5e5ba43071d6151b3d881deb9300d000e"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def production_backends(build: Path) -> dict[str, str]:
    cache = build / "llama" / "src" / "ext_llama-build" / "CMakeCache.txt"
    values: dict[str, str] = {}
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = None
            for key in ("GGML_CUDA", "GGML_VULKAN", "CMAKE_CUDA_ARCHITECTURES",
                        "CMAKE_CUDA_COMPILER", "GGML_CUDA_FAST_MATH",
                        "GGML_CUDA_GRAPHS", "GGML_NATIVE", "GGML_AVX",
                        "GGML_AVX2", "GGML_AVX512", "GGML_FMA", "GGML_F16C",
                        "GGML_BMI2"):
                if line.startswith(key + ":"):
                    m = (key, line.split("=", 1)[1].strip())
                    break
            if m:
                values.setdefault(m[0], m[1])
    return values


def fetch_archive(url: str, dest: Path) -> Path:
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[vlx-upstream] downloading {url}")
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", type=Path, required=True,
                        help="production build whose backend set to mirror")
    parser.add_argument("--archive", type=Path,
                        help="pinned upstream repo archive "
                             "(default: cached download)")
    parser.add_argument("--llama-archive", type=Path,
                        help="pinned upstream llama.cpp submodule archive")
    parser.add_argument("--out", type=Path,
                        help="reference tree (default: <build>/Testing/"
                             "framework-references/vlx-upstream)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--backend", choices=("production", "cpu"),
                        default="production")
    args = parser.parse_args()

    out = args.out or (args.build / "Testing" / "framework-references" /
                       "vlx-upstream")
    src = out / "src"
    marker = out / "READY"
    bin_dir = None
    for candidate in (src / "cpp_ggml" / "build", src / "cpp_ggml" / "build"):
        exe = candidate / ("vlx-cli.exe" if os.name == "nt" else "vlx-cli")
        if exe.is_file():
            bin_dir = candidate
            break
    if marker.is_file() and bin_dir:
        print(bin_dir)
        return 0

    import shutil
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)

    repo_archive = args.archive
    if repo_archive is None:
        repo_archive = out / "downloads" / f"vlx-seek-ggml-{UPSTREAM_REVISION}.tar.gz"
    fetch_archive(
        f"{UPSTREAM_REPO}/archive/{UPSTREAM_REVISION}.tar.gz", repo_archive)
    llama_archive = args.llama_archive
    if llama_archive is None:
        llama_archive = (out / "downloads" /
                         f"llama-cpp-{UPSTREAM_LLAMA_SUBMODULE}.tar.gz")
    fetch_archive(
        "https://github.com/ggml-org/llama.cpp/archive/"
        f"{UPSTREAM_LLAMA_SUBMODULE}.tar.gz", llama_archive)

    print(f"[vlx-upstream] repo sha256 {sha256(repo_archive)}")
    print(f"[vlx-upstream] llama submodule sha256 {sha256(llama_archive)}")
    with tarfile.open(repo_archive) as tf:
        tf.extractall(src)
    extracted = next(src.glob("VLX-Seek-GGML-*"))
    extracted.rename(src / "repo")
    sub = src / "repo" / "cpp_ggml" / "third_party" / "llama.cpp"
    with tarfile.open(llama_archive) as tf:
        tf.extractall(sub.parent)
    llama_extracted = next(sub.parent.glob("llama.cpp-*"))
    llama_extracted.rename(sub)

    # Pre-apply the upstream's own llama.cpp patch and write its replay stamp
    # so the upstream CMake patch step skips (its replay refuses a dirty
    # extracted tree with no stamp; a pristine pre-apply is byte-identical).
    import glob
    patches = sorted(glob.glob(str(src / "repo" / "cpp_ggml" / "third_party" /
                                "llama-patches" / "*.patch")))
    stamp_lines = []
    for patch in patches:
        subprocess.run(["patch", "-p1", "--forward", "--silent",
                        "--directory", str(sub), "-i", patch], check=True)
        stamp_lines.append(
            f"{Path(patch).name} {sha256(Path(patch))}\n")
    (sub / ".vlx-patches-applied").write_text("".join(stamp_lines))

    build_dir = src / "repo" / "cpp_ggml" / "build"
    cmake_args = [
        "cmake", "-S", str(src / "repo" / "cpp_ggml"), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    backends = production_backends(args.build)
    if args.backend == "production":
        for key, value in backends.items():
            cmake_args.append(f"-D{key}={value}")
    subprocess.run(cmake_args, check=True)
    subprocess.run(["cmake", "--build", str(build_dir),
                    "--target", "vlx-cli",
                    "-j", str(args.jobs)], check=True)
    marker.write_text(f"repo={UPSTREAM_REVISION}\n"
                      f"llama={UPSTREAM_LLAMA_SUBMODULE}\n"
                      f"repo_archive_sha256={sha256(repo_archive)}\n"
                      f"llama_archive_sha256={sha256(llama_archive)}\n")
    print(build_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
