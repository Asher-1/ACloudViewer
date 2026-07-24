#!/usr/bin/env python3
"""
Patch ggml CPU ALL_VARIANTS x86 backend list for compiler-adaptive builds.

Problem: upstream ggml always registers cooperlake/zen4/sapphirerapids when
GGML_CPU_ALL_VARIANTS=ON on non-MSVC toolchains, even when the compiler lacks
-mavx512bf16 / -mamx-tile / -mamx-int8 (e.g. GCC 9 on Ubuntu 20.04).

Fix: gate BF16 and AMX variants behind check_cxx_compiler_flag(), matching the
existing POWER11 adaptive pattern in ggml src/CMakeLists.txt.

Idempotent: skips when the adaptive block is already present.
Usage: python3 apply_cpu_all_variants_compiler_checks.py <ggml_source_dir>
"""

import sys
import os

MARKER = "GGML_CXX_SUPPORTS_AVX512_BF16"
MARKER_AVX_VNNI = "GGML_CXX_SUPPORTS_AVX_VNNI"
OLD_BLOCK = """        if (NOT MSVC)
            # MSVC 2022 doesn't support BF16 intrinsics without `/arch:AVX10.1` ?!
            # https://learn.microsoft.com/en-us/cpp/intrinsics/x64-amd64-intrinsics-list?view=msvc-170
            # https://learn.microsoft.com/en-us/cpp/build/reference/arch-x64?view=msvc-170
            ggml_add_cpu_backend_variant(cooperlake     SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VNNI AVX512_BF16)
            ggml_add_cpu_backend_variant(zen4           SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16)
        endif()
        ggml_add_cpu_backend_variant(alderlake          SSE42 AVX F16C FMA AVX2 BMI2 AVX_VNNI)
        if (NOT MSVC)
            # MSVC doesn't support AMX
            ggml_add_cpu_backend_variant(sapphirerapids SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 AMX_TILE AMX_INT8)
        endif()"""

NEW_BLOCK = """        if (NOT MSVC)
            # MSVC 2022 doesn't support BF16 intrinsics without `/arch:AVX10.1` ?!
            # https://learn.microsoft.com/en-us/cpp/intrinsics/x64-amd64-intrinsics-list?view=msvc-170
            # https://learn.microsoft.com/en-us/cpp/build/reference/arch-x64?view=msvc-170
            check_cxx_compiler_flag("-mavx512bf16" GGML_CXX_SUPPORTS_AVX512_BF16)
            if (GGML_CXX_SUPPORTS_AVX512_BF16)
                ggml_add_cpu_backend_variant(cooperlake     SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VNNI AVX512_BF16)
                ggml_add_cpu_backend_variant(zen4           SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16)
            else()
                message(STATUS "Skipping cooperlake/zen4 CPU backends: compiler lacks -mavx512bf16")
            endif()
        endif()
        ggml_add_cpu_backend_variant(alderlake          SSE42 AVX F16C FMA AVX2 BMI2 AVX_VNNI)
        if (NOT MSVC)
            # MSVC doesn't support AMX
            if (NOT DEFINED GGML_CXX_SUPPORTS_AVX512_BF16)
                check_cxx_compiler_flag("-mavx512bf16" GGML_CXX_SUPPORTS_AVX512_BF16)
            endif()
            check_cxx_compiler_flag("-mamx-tile" GGML_CXX_SUPPORTS_AMX_TILE)
            check_cxx_compiler_flag("-mamx-int8" GGML_CXX_SUPPORTS_AMX_INT8)
            if (GGML_CXX_SUPPORTS_AVX512_BF16 AND GGML_CXX_SUPPORTS_AMX_TILE AND GGML_CXX_SUPPORTS_AMX_INT8)
                ggml_add_cpu_backend_variant(sapphirerapids SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 AMX_TILE AMX_INT8)
            else()
                message(STATUS "Skipping sapphirerapids CPU backend: compiler lacks AMX/BF16 flags")
            endif()
        endif()"""


def _patch_alderlake_gate(content):
    if MARKER_AVX_VNNI in content:
        return content, False

    old_alderlake = """        ggml_add_cpu_backend_variant(alderlake          SSE42 AVX F16C FMA AVX2 BMI2 AVX_VNNI)
        if (NOT MSVC)
            # MSVC doesn't support AMX"""

    new_alderlake = """        if (NOT MSVC)
            check_cxx_compiler_flag("-mavxvnni" GGML_CXX_SUPPORTS_AVX_VNNI)
            if (GGML_CXX_SUPPORTS_AVX_VNNI)
                ggml_add_cpu_backend_variant(alderlake          SSE42 AVX F16C FMA AVX2 BMI2 AVX_VNNI)
            else()
                message(STATUS "Skipping alderlake CPU backend: compiler lacks -mavxvnni")
            endif()
        else()
            ggml_add_cpu_backend_variant(alderlake          SSE42 AVX F16C FMA AVX2 BMI2 AVX_VNNI)
        endif()
        if (NOT MSVC)
            # MSVC doesn't support AMX"""

    if old_alderlake not in content:
        print("[ggml-patch] alderlake CPU backend block not found; ggml version may differ")
        return content, False

    return content.replace(old_alderlake, new_alderlake, 1), True


def patch_cpu_variants(src_dir):
    cmake_path = os.path.join(src_dir, "src", "CMakeLists.txt")
    if not os.path.exists(cmake_path):
        print(f"[ggml-patch] CMakeLists not found: {cmake_path}")
        return False

    with open(cmake_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    if MARKER in content:
        print("[ggml-patch] CPU ALL_VARIANTS BF16/AMX checks already applied")
    elif OLD_BLOCK not in content:
        print("[ggml-patch] CPU ALL_VARIANTS block not found; ggml version may differ")
        return False
    else:
        content = content.replace(OLD_BLOCK, NEW_BLOCK, 1)
        print("[ggml-patch] Applied CPU ALL_VARIANTS BF16/AMX compiler checks")

    content, applied_alderlake = _patch_alderlake_gate(content)
    if applied_alderlake:
        print("[ggml-patch] Applied alderlake AVX-VNNI compiler check")

    with open(cmake_path, "w", encoding="utf-8") as handle:
        handle.write(content)

    if MARKER in content or applied_alderlake:
        return True

    return False


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <ggml_source_dir>", file=sys.stderr)
        sys.exit(1)
    if not patch_cpu_variants(sys.argv[1]):
        sys.exit(1)


if __name__ == "__main__":
    main()
