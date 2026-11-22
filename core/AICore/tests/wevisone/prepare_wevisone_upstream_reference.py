#!/usr/bin/env python3
"""Build the upstream WeVisOne-VL-GGML reference CLI for the wevisone probe.

The wevisone task's parity reference is the upstream Asher-1/WeVisOne-VL-GGML
C++ port at a pinned revision: ``test_wevisone_probe --upstream-dir DIR`` runs
``wevisone-cli run`` (and ``batch``) from DIR on the same gguf files, image,
task, query and device and requires identical greedy answers / matching boxes
and mattes. That CLI must come from the *pinned upstream revision* without
AICore's task tree, so this script:

1. downloads (or verifies ``--archive`` of) the pinned upstream repository
   archive plus its pinned llama.cpp submodule archive;
2. extracts both pristine into ``<out>/src``;
3. configures one build directory per GPU backend (the upstream CMake
   enforces at most one GPU backend per build directory, unlike the vlx
   reference) mirroring the production ext_llama cache flags, CPU native;
4. builds ``wevisone-cli`` and prints the binary directory.

The upstream CMake unconditionally declares a ``wevisone-capi-smoke`` target
whose source lives under the gitignored ``cpp_ggml/tests/`` directory, so a
pristine revision archive cannot configure without it; a placeholder main is
generated for that non-reference target only (the parity binary is
``wevisone-cli``, built from tracked sources).

Pass the printed directory to validate_all.py as
``--reference wevisone_upstream=<bin dir>`` (run once per backend with
``--gpu-backend cuda|vulkan``). Nothing generated here is checked in.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from runtime_environment import executable_name

REPO = Path(__file__).resolve().parents[4]

# Pinned upstream integration revision (the C++/GGML port commit) and its
# llama.cpp submodule commit (.gitmodules gitlink).
UPSTREAM_REPO = "https://github.com/Asher-1/WeVisOne-VL-GGML"
UPSTREAM_REVISION = "ab31d9de796b1d7f108f8a541227a66150e90db3"
UPSTREAM_LLAMA_SUBMODULE = "620d2225cc436ab0b001e3200e590e3b286ef9ab"

# Placeholder for the gitignored upstream smoke-test source; the target is
# never built by this script (only wevisone-cli is), it only has to configure.
SMOKE_STUB = (
    "/* Generated placeholder: upstream gitignores cpp_ggml/tests/, so the\n"
    " * pinned revision archive lacks wevisone_capi_smoke.c while its\n"
    " * CMakeLists declares the target unconditionally. Not the parity\n"
    " * reference; wevisone-cli is built from tracked sources only. */\n"
    "int main(void) { return 0; }\n")


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
    print(f"[wevisone-upstream] downloading {url}")
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
                             "framework-references/wevisone-upstream)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--backend", choices=("production", "cpu"),
                        default="production")
    parser.add_argument("--gpu-backend", choices=("cuda", "vulkan"),
                        default="cuda",
                        help="which GPU backend's reference build to produce "
                             "(upstream allows one GPU backend per build "
                             "directory; ignored with --backend cpu)")
    args = parser.parse_args()

    out = args.out or (args.build / "Testing" / "framework-references" /
                       "wevisone-upstream")
    src = out / "src"
    suffix = args.gpu_backend if args.backend == "production" else "cpu"
    build_dir = src / "repo" / "cpp_ggml" / f"build-{suffix}"
    marker = out / f"READY-{suffix}"
    exe = build_dir / executable_name("wevisone-cli")
    if marker.is_file() and exe.is_file():
        print(build_dir)
        return 0

    import shutil
    src_stamp = src / "repo" / ".wwv-reference-revision"
    if not (src_stamp.is_file()
            and src_stamp.read_text().strip() == UPSTREAM_REVISION):
        if src.exists():
            shutil.rmtree(src)
        src.mkdir(parents=True)

        repo_archive = args.archive
        if repo_archive is None:
            repo_archive = (out / "downloads" /
                            f"wevisone-vl-ggml-{UPSTREAM_REVISION}.tar.gz")
        fetch_archive(
            f"{UPSTREAM_REPO}/archive/{UPSTREAM_REVISION}.tar.gz", repo_archive)
        llama_archive = args.llama_archive
        if llama_archive is None:
            llama_archive = (out / "downloads" /
                             f"llama-cpp-{UPSTREAM_LLAMA_SUBMODULE}.tar.gz")
        fetch_archive(
            "https://github.com/ggml-org/llama.cpp/archive/"
            f"{UPSTREAM_LLAMA_SUBMODULE}.tar.gz", llama_archive)

        print(f"[wevisone-upstream] repo sha256 {sha256(repo_archive)}")
        print(f"[wevisone-upstream] llama submodule sha256 "
              f"{sha256(llama_archive)}")
        with tarfile.open(repo_archive) as tf:
            tf.extractall(src)
        extracted = next(src.glob("WeVisOne-VL-GGML-*"))
        extracted.rename(src / "repo")
        sub = src / "repo" / "cpp_ggml" / "third_party" / "llama.cpp"
        with tarfile.open(llama_archive) as tf:
            tf.extractall(sub.parent)
        llama_extracted = next(sub.parent.glob("llama.cpp-*"))
        llama_extracted.rename(sub)

        # Pre-apply the upstream's own llama.cpp patch and write its replay
        # stamp so the upstream CMake patch step skips (its replay refuses a
        # dirty extracted tree with no stamp; a pristine pre-apply is
        # byte-identical). Stamp format: cmake/apply_patches.cmake expects
        # "<name> <sha256>\n".
        import glob
        patches = sorted(glob.glob(str(src / "repo" / "cpp_ggml" /
                                    "third_party" / "llama-patches" /
                                    "*.patch")))
        stamp_lines = []
        for patch in patches:
            # patch chdirs into --directory before opening -i, so both paths
            # must be absolute.
            subprocess.run(["patch", "-p1", "--forward", "--silent",
                            "--directory", str(sub.resolve()),
                            "-i", str(Path(patch).resolve())], check=True)
            stamp_lines.append(
                f"{Path(patch).name} {sha256(Path(patch))}\n")
        (sub / ".wwv-patches-applied").write_text("".join(stamp_lines))

        smoke = src / "repo" / "cpp_ggml" / "tests" / "wevisone_capi_smoke.c"
        smoke.parent.mkdir(parents=True, exist_ok=True)
        smoke.write_text(SMOKE_STUB)
        src_stamp.write_text(UPSTREAM_REVISION + "\n")

    cmake_args = [
        "cmake", "-S", str(src / "repo" / "cpp_ggml"), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if args.backend == "production":
        for key, value in production_backends(args.build).items():
            if key in ("GGML_CUDA", "GGML_VULKAN"):
                # Upstream gates the GPU backends through its own options
                # (one GPU backend per build directory).
                continue
            cmake_args.append(f"-D{key}={value}")
        cmake_args.append(f"-DWWV_GGML_{args.gpu_backend.upper()}=ON")
    subprocess.run(cmake_args, check=True)
    subprocess.run(["cmake", "--build", str(build_dir),
                    "--target", "wevisone-cli",
                    "-j", str(args.jobs)], check=True)
    marker.write_text(f"repo={UPSTREAM_REVISION}\n"
                      f"llama={UPSTREAM_LLAMA_SUBMODULE}\n"
                      f"gpu_backend={suffix}\n")
    print(build_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())