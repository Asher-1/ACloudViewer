#!/usr/bin/env python3
"""Build an isolated upstream CPU baseline with the candidate's compiler flags.

Only CPU module discovery/thread configuration is adapted. Graph operations,
image preprocessing and output selection remain the pinned upstream source.
The original checkout and its builds are never modified.
"""

import argparse
import difflib
import json
from pathlib import Path
import subprocess
import tempfile

from demo_reference_probe import REVISION, sha256
from upstream_metrics_probe import build_settings, candidate_build_settings


GGML_REVISION = "8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b"


def adapt_cpu_loading(source, modules):
    replacements = (
        ("ggml_backend_cpu_init()",
         "ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)", 2),
        ("ggml_backend_cpu_set_n_threads(", "set_cpu_threads(", 2),
        ("        gguf_init_params params{};",
         "        ggml_backend_load_all_from_path(" + json.dumps(str(modules)) +
         ");\n        gguf_init_params params{};", 1),
        ("static void ggml_log(",
         "static void set_cpu_threads(ggml_backend_t backend, int threads) {\n"
         "    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));\n"
         "    auto setter = reinterpret_cast<ggml_backend_set_n_threads_t>(\n"
         "        ggml_backend_reg_get_proc_address(reg, \"ggml_backend_set_n_threads\"));\n"
         "    require(setter != nullptr, \"CPU thread setter unavailable\");\n"
         "    setter(backend, threads);\n"
         "}\n\nstatic void ggml_log(", 1),
    )
    for old, new, count in replacements:
        if source.count(old) != count:
            raise RuntimeError(f"upstream CPU adapter expected {count} occurrences of {old}")
        source = source.replace(old, new)
    return source


def clone_pinned(original, destination, revision):
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout",
                    str(original), str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--quiet",
                    "--detach", revision], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    upstream = args.upstream.resolve()
    settings = candidate_build_settings(args.candidate_build.resolve())
    if settings.get("GGML_BACKEND_DL") != "ON" or settings.get("GGML_CPU_ALL_VARIANTS") != "ON":
        parser.error("candidate must use dynamic CPU modules and GGML_CPU_ALL_VARIANTS")
    args.output_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="ram-upstream-cpu-", dir=args.output_root)).resolve()
    source, build = root / "source", root / "build"
    clone_pinned(upstream, source, REVISION)
    ggml = source / "cpp_ggml/third_party/ggml"
    clone_pinned(upstream / "cpp_ggml/third_party/ggml", ggml, GGML_REVISION)
    # This compiler capability patch changes which modules can be built, not
    # their arithmetic. Upstream CMake owns and applies its arithmetic patches.
    patch = Path(__file__).resolve().parents[4] / (
        "3rdparty/llama/ggml-patches/cpu_all_variants/0001-cpu-all-variants-compiler-checks.patch")
    subprocess.run(["git", "-C", str(ggml), "apply", str(patch)], check=True)
    main_cpp = source / "cpp_ggml/src/main.cpp"
    original = main_cpp.read_text()
    adapted = adapt_cpu_loading(original, build / "bin")
    main_cpp.write_text(adapted)
    (root / "cpu-loader-adapter.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), adapted.splitlines(keepends=True),
        fromfile="a/cpp_ggml/src/main.cpp", tofile="b/cpp_ggml/src/main.cpp")))
    keys = ("CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER",
            "CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS", "CMAKE_C_FLAGS_RELEASE",
            "CMAKE_CXX_FLAGS_RELEASE", "GGML_NATIVE", "GGML_BACKEND_DL",
            "GGML_CPU_ALL_VARIANTS", "GGML_CPU_REPACK", "GGML_OPENMP", "GGML_LLAMAFILE")
    command = ["cmake", "-S", str(source / "cpp_ggml"), "-B", str(build),
               "-DBUILD_SHARED_LIBS=ON", "-DRAM_GGML_CUDA=OFF", "-DRAM_GGML_VULKAN=OFF"]
    command.extend(f"-D{key}={settings[key]}" for key in keys if key in settings)
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", str(args.jobs)], check=True)
    (root / "provenance.json").write_text(json.dumps({
        "upstream_revision": REVISION, "ggml_revision": GGML_REVISION,
        "compiler_capability_patch_sha256": sha256(patch),
        "compiled_source_sha256": sha256(main_cpp),
        "loader_adapter_sha256": sha256(root / "cpu-loader-adapter.patch"),
        "settings": build_settings(build), "configure_command": command,
    }, indent=2) + "\n")
    print(f"AICORE_TEST_RAM_UPSTREAM_BUILD={build}")


if __name__ == "__main__":
    main()
