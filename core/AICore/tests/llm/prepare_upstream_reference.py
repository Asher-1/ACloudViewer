#!/usr/bin/env python3
"""Build the upstream llama.cpp reference tools for the llm task probe.

The llm task's truth source is llama.cpp itself: ``test_llm_probe
--upstream-dir DIR`` runs ``llama-mtmd-cli`` / ``llama-completion`` from DIR on
the same model, projector, image, prompt and device and requires identical
greedy text. Those tools must come from the *same pinned archive* that AICore
embeds, but *without* the AICore patch chain, so this script:

1. reads LLAMA_REVISION / LLAMA_SHA256 from 3rdparty/llama/llama_cpp.cmake;
2. verifies the archive (the cached download, or ``--archive``);
3. extracts it pristine into ``<out>/src``;
4. configures the backend set of the production build (CUDA / Vulkan and the
   CUDA architectures read from its ext_llama CMakeCache), CPU native;
5. builds ``llama-mtmd-cli`` and ``llama-completion`` and prints the bin dir.

Pass the printed directory to validate_all.py as
``--reference llama_upstream=<bin dir>``. Nothing generated here is checked in.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OWNER = REPO / "3rdparty" / "llama" / "llama_cpp.cmake"


def pinned() -> tuple[str, str]:
    text = OWNER.read_text(encoding="utf-8")
    revision = re.search(r'set\(LLAMA_REVISION "([0-9a-f]{40})"\)', text)
    digest = re.search(r'set\(LLAMA_SHA256 "([0-9a-f]{64})"\)', text)
    if not revision or not digest:
        sys.exit(f"cannot read the llama pin from {OWNER}")
    return revision.group(1), digest.group(1)


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
            # Compiler, architecture and arithmetic switches jointly determine reference behavior;
            # copying only backend switches lets CMake reselect host toolchains and invalidates controlled comparison.
            m = re.match(r"^(GGML_CUDA|GGML_VULKAN|CMAKE_CUDA_ARCHITECTURES|"
                         r"CMAKE_CUDA_COMPILER|GGML_CUDA_FAST_MATH|"
                         r"GGML_CUDA_GRAPHS|GGML_CPU_ALL_VARIANTS|GGML_NATIVE|GGML_BACKEND_DL|"
                                                  r"GGML_AVX|GGML_AVX2|GGML_AVX512|GGML_FMA|GGML_F16C|GGML_BMI2)"
                         r":[A-Z]+=(.*)$", line)
            if m:
                values[m.group(1)] = m.group(2)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", type=Path, required=True,
                        help="production build whose backend set to mirror")
    parser.add_argument("--archive", type=Path,
                        help="pinned llama.cpp archive (default: cached download)")
    parser.add_argument("--out", type=Path,
                        help="reference tree (default: <build>/Testing/"
                             "framework-references/llama-upstream)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--backend", choices=("production", "cpu"), default="production",
                        help="Reference device scope; cpu retains production ISA/arithmetic without building GPU toolchains")
    args = parser.parse_args()

    revision, digest = pinned()
    archive = args.archive or (REPO / "3rdparty_downloads" / "llama" /
                               f"{revision}.tar.gz")
    if not archive.is_file():
        sys.exit(f"pinned archive missing: {archive} (configure the production "
                 "build once, or pass --archive)")
    if sha256(archive) != digest:
        sys.exit(f"archive digest mismatch for {archive}")
    default_name = "llama-upstream-cpu" if args.backend == "cpu" else "llama-upstream"
    out = (args.out or args.build / "Testing" / "framework-references" /
           default_name).absolute()
    src = out / "src"
    stamp = out / "revision.txt"
    if not stamp.is_file() or stamp.read_text().strip() != revision:
        src.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts[1:]
                if not parts:
                    continue
                member.name = str(Path(*parts))
                tar.extract(member, src)
        stamp.write_text(revision + "\n")

    backends = production_backends(args.build)
    if args.backend == "cpu":
        # Narrow only reference device scope, preserving selected CPU ISA/arithmetic and production builds.
        backends.update(GGML_CUDA="OFF", GGML_VULKAN="OFF")
    on = lambda key: backends.get(key, "OFF").upper() in ("ON", "1", "TRUE")
    configure = [
        "cmake", "-S", str(src), "-B", str(out / "build"),
        "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON",
        # Pristine lacks the production compiler-feature guard; old GCC cannot
        # build all BF16/AMX variants. Build only the current host's CPU ISA for the reference and
        # copy explicit ISA switches; qualify production all-variant deployment independently.
        "-DGGML_CPU_ALL_VARIANTS=OFF",
        "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_OPENSSL=OFF",
        f"-DGGML_CUDA={'ON' if on('GGML_CUDA') else 'OFF'}",
        f"-DGGML_VULKAN={'ON' if on('GGML_VULKAN') else 'OFF'}",
    ]
    for key in ("CMAKE_CUDA_ARCHITECTURES", "CMAKE_CUDA_COMPILER",
                "GGML_CUDA_FAST_MATH", "GGML_CUDA_GRAPHS",
                "GGML_NATIVE", "GGML_BACKEND_DL", "GGML_AVX", "GGML_AVX2",
                "GGML_AVX512", "GGML_FMA", "GGML_F16C", "GGML_BMI2"):
        if key in backends and (on("GGML_CUDA") or "CUDA" not in key):
            configure.append(f"-D{key}={backends[key]}")
    subprocess.run(configure, check=True)
    subprocess.run(["cmake", "--build", str(out / "build"), "--config",
                    "Release", "-j", str(args.jobs), "--target",
                    "llama-mtmd-cli", "llama-completion"], check=True)
    print(out / "build" / "bin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
