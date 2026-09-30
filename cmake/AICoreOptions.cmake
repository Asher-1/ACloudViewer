# AICoreOptions.cmake — single entry for all public AICore_* CMake options
#
# Policy: every user-facing AICore switch lives here (core toggle, tests, backends,
# packaging). ggml.cmake keeps GGML_* internally; this file maps between them.
# Direct -DGGML_* on the command line is ignored (cleared with a warning);
# stale GGML_* entries left in CMakeCache from older configures are cleared the
# same way before syncing from AICore_*.
#
# Root CMakeLists.txt: include(this file) then nothing else AICore-related.
#
# When adding a new option:
#   1. option(AICore_* ...) here with a clear description
#   2. aicore_sync_options_to_ggml() — forward to GGML_* before ggml.cmake
#   3. aicore_sync_results_from_ggml() — mirror read-only GGML_*_ENABLED if needed
#   4. BUILD.md option table + util/ci_utils.{sh,ps1} + compile guides
#   5. Do NOT document GGML_* as user-facing CMake flags
#
# See core/AICore/README.md § CMake options.

# --- core toggle + tests (must be declared before find_dependencies / add_subdirectory) ---
option(AICore_ENABLED
    "Build unified AI inference core (libAICore.so; auto-enables GGML)"
    OFF)
option(AICore_BUILD_TESTS
    "Build AICore contract and inference tests (public ABI/runtime suite)"
    OFF)
option(AICore_BUILD_WHITEBOX_TESTS
    "Build AICore private implementation tests (requires AICore_BUILD_TESTS)"
    OFF)

if(AICore_BUILD_WHITEBOX_TESTS AND NOT AICore_BUILD_TESTS)
    message(FATAL_ERROR
        "AICore_BUILD_WHITEBOX_TESTS=ON requires AICore_BUILD_TESTS=ON")
endif()

if(AICore_ENABLED)
    set(GGML_ENABLED ON CACHE BOOL "Internal: synced from AICore_ENABLED" FORCE)
else()
    set(GGML_ENABLED OFF CACHE BOOL "Internal: synced from AICore_ENABLED" FORCE)
endif()
mark_as_advanced(GGML_ENABLED)

if(AICore_BUILD_TESTS)
    enable_testing()
endif()

# --- inference backends + packaging ---
if(APPLE)
    set(_AICORE_USE_METAL_DEFAULT ON)
    set(_AICORE_USE_VULKAN_DEFAULT OFF)
elseif(WIN32 OR UNIX)
    set(_AICORE_USE_METAL_DEFAULT OFF)
    set(_AICORE_USE_VULKAN_DEFAULT ON)
else()
    set(_AICORE_USE_METAL_DEFAULT OFF)
    set(_AICORE_USE_VULKAN_DEFAULT OFF)
endif()

option(AICore_USE_METAL
    "Build AICore Metal inference backend (Apple default ON; non-Apple ignored)"
    ${_AICORE_USE_METAL_DEFAULT})
option(AICore_USE_VULKAN
    "Build AICore Vulkan inference backend (Linux/Windows default ON; macOS OFF)"
    ${_AICORE_USE_VULKAN_DEFAULT})
option(AICore_USE_CUDA
    "Build AICore CUDA inference backend (developer opt-in; not BUILD_CUDA_MODULE)"
    OFF)
option(AICore_USE_SYCL
    "Build AICore SYCL inference backend (Intel oneAPI; developer opt-in)"
    OFF)
option(AICore_SYCL_USE_DNN
    "Enable oneDNN kernels in AICore SYCL backend (requires AICore_USE_SYCL=ON)"
    ON)
option(AICore_USE_OPENCL
    "Build AICore OpenCL inference backend (legacy/Adreno developer opt-in)"
    OFF)
option(AICore_BUNDLE_CUDA_RUNTIME
    "Redist CUDA runtime libs into installer for driver-only targets (requires AICore_USE_CUDA=ON).\
Note: libggml-cuda.so always depends on libcublas.so.* (ggml hardwires non-quantized\
matmul to cuBLAS), so driver-only Linux/Windows installers need this ON to ship\
libcublas.so.* next to the app; without it the CUDA backend silently fails to load\
on machines that only have the NVIDIA driver installed."
    OFF)
option(AICore_CUDA_FORCE_MMQ
    "Force ggml-cuda MMQ kernels instead of cuBLAS for quantized matmul.\
OFF (default): performance-parity with upstream ggml/ultralytics-ggml builds\
(cuBLAS stays linked, so libggml-cuda.so has a libcublas DT_NEEDED — driver-only\
deployments must also set AICore_BUNDLE_CUDA_RUNTIME=ON).\
ON: self-contained libggml-cuda.so (no libcudart/libcublas DT_NEEDED) at the\
cost of slower small conv/matmul kernels (see core/AICore/tests/TESTING.md)."
    OFF)
set(AICore_OPENCL_TARGET_VERSION "200" CACHE STRING
    "OpenCL host API target for AICore (120, 200, or 300)")
set_property(CACHE AICore_OPENCL_TARGET_VERSION PROPERTY STRINGS 120 200 300)
option(AICore_CPU_ALL_VARIANTS
    "Build all ggml CPU micro-architecture backend variants (llama.cpp release style; +~15MB on x86)"
    OFF)

# --- task-private build wiring (declared here, wired by the owning task) ---
# These options belong to one task under core/AICore/src/tasks/; the wiring
# itself lives next to the task (sam3d/cmake/pbr.cmake,
# trellis/cmake/cumesh.cmake) and is included from
# core/AICore/CMakeLists.txt. They are DECLARED here —
# before find_dependencies.cmake — so the bundled OpenCV build
# (3rdparty/opencv/opencv_build.cmake) can wire its photo module from the
# SAM3D PBR switch; core/AICore keeps only the cross-module CUDA linkage
# check. Keep option names/strings in sync with the docs (BUILD.md; CuMesh
# rows live in core/AICore/docs/ARCHITECTURE.md).
#
# PLUGIN_STANDARD_QSAM3D is pre-declared here for the same reason: the PBR
# auto-enable below reads it, and the photo wiring inherits the result —
# both run before plugins/core/Standard/qSAM3D/CMakeLists.txt (root L800,
# after core) gets to declare it. That plugin-side option() then no-ops on
# this cache entry — keep the docstring identical.
option( PLUGIN_STANDARD_QSAM3D "Install qSAM3D (SAM 3D Objects image-to-3D GGML) plugin" OFF )

# SAM3D PBR auto-enable: CUDA + the qSAM3D plugin. An explicit
# -DAICore_USE_SAM3D_PBR=ON/OFF always wins (option() never overwrites an
# existing cache entry). 3rdparty_vtk availability cannot be known this
# early (the VTK section of find_dependencies runs later) — pbr.cmake
# degrades with a warning (CuMesh-style ladder) when the target was never
# created, so the auto-enabled default stays build-safe.
set(_sam3d_pbr_auto OFF)
if(AICore_USE_CUDA AND PLUGIN_STANDARD_QSAM3D)
    set(_sam3d_pbr_auto ON)
endif()
option(AICore_USE_SAM3D_PBR
    "SAM3D: official native PBR pipeline (CUDA-only; auto-ON with AICore_USE_CUDA=ON + PLUGIN_STANDARD_QSAM3D=ON; opt-in vendors: nvdiffrast NC, mip-splatting NC, pymeshfix GPL)"
    ${_sam3d_pbr_auto})
option(AICore_USE_SAM3D_PBR_NVDIFFRAST
    "SAM3D PBR: compile+link the vendored nvdiffrast 253ac4f cudaraster core (NVIDIA non-commercial license acknowledgment)"
    ON)
option(AICore_USE_SAM3D_PBR_MESHFIX
    "SAM3D PBR: compile+link the vendored pymeshfix 79adef9 TMesh boundary repair (GPL-3.0-or-commercial license acknowledgment)"
    ON)
option(AICore_USE_CUMESH
    "Trellis: CuMesh GPU chart clustering for the UV-atlas bake (default ON; auto-disabled with a warning when CUDA/libtorch/Python headers are unavailable)"
    ON)

# Read-only configure results: set only in aicore_sync_results_from_ggml() as CACHE
# entries. Do NOT initialize normal variables here — they would shadow the cache
# and break ${AICore_*_ENABLED} reads (e.g. summary showing OFF while ggml built Vulkan).

function(_aicore_clear_stale_ggml_cache ggml_var aicore_var)
    if(NOT DEFINED CACHE{${ggml_var}})
        return()
    endif()
    get_property(_help CACHE ${ggml_var} PROPERTY HELPSTRING)
    if(_help MATCHES "Internal: synced from")
        return()
    endif()
    if(NOT "${${ggml_var}}" STREQUAL "${${aicore_var}}")
        message(WARNING
            "Ignoring stale CMake cache ${ggml_var}=${${ggml_var}}; "
            "using ${aicore_var}=${${aicore_var}} instead. "
            "Run: cmake -U ${ggml_var} (or delete CMakeCache.txt) to silence.")
    endif()
    unset(${ggml_var} CACHE)
endfunction()

function(aicore_sync_options_to_ggml)
    _aicore_clear_stale_ggml_cache(GGML_ENABLED AICore_ENABLED)
    _aicore_clear_stale_ggml_cache(GGML_USE_METAL AICore_USE_METAL)
    _aicore_clear_stale_ggml_cache(GGML_USE_VULKAN AICore_USE_VULKAN)
    _aicore_clear_stale_ggml_cache(GGML_USE_CUDA AICore_USE_CUDA)
    _aicore_clear_stale_ggml_cache(GGML_USE_SYCL AICore_USE_SYCL)
    _aicore_clear_stale_ggml_cache(GGML_SYCL_USE_DNN AICore_SYCL_USE_DNN)
    _aicore_clear_stale_ggml_cache(GGML_USE_OPENCL AICore_USE_OPENCL)
    _aicore_clear_stale_ggml_cache(GGML_BUNDLE_CUDA_RUNTIME AICore_BUNDLE_CUDA_RUNTIME)
    _aicore_clear_stale_ggml_cache(GGML_CUDA_FORCE_MMQ AICore_CUDA_FORCE_MMQ)
    _aicore_clear_stale_ggml_cache(GGML_OPENCL_TARGET_VERSION AICore_OPENCL_TARGET_VERSION)
    _aicore_clear_stale_ggml_cache(GGML_CPU_ALL_VARIANTS AICore_CPU_ALL_VARIANTS)

    if(AICore_BUNDLE_CUDA_RUNTIME AND NOT AICore_USE_CUDA)
        message(FATAL_ERROR
            "AICore_BUNDLE_CUDA_RUNTIME=ON requires AICore_USE_CUDA=ON")
    endif()
    if(NOT AICore_OPENCL_TARGET_VERSION MATCHES "^(120|200|300)$")
        message(FATAL_ERROR
            "AICore_OPENCL_TARGET_VERSION must be one of 120, 200, or 300")
    endif()

    set(GGML_USE_METAL ${AICore_USE_METAL} CACHE BOOL
        "Internal: synced from AICore_USE_METAL" FORCE)
    set(GGML_USE_VULKAN ${AICore_USE_VULKAN} CACHE BOOL
        "Internal: synced from AICore_USE_VULKAN" FORCE)
    set(GGML_USE_CUDA ${AICore_USE_CUDA} CACHE BOOL
        "Internal: synced from AICore_USE_CUDA" FORCE)
    set(GGML_USE_SYCL ${AICore_USE_SYCL} CACHE BOOL
        "Internal: synced from AICore_USE_SYCL" FORCE)
    set(GGML_SYCL_USE_DNN ${AICore_SYCL_USE_DNN} CACHE BOOL
        "Internal: synced from AICore_SYCL_USE_DNN" FORCE)
    set(GGML_USE_OPENCL ${AICore_USE_OPENCL} CACHE BOOL
        "Internal: synced from AICore_USE_OPENCL" FORCE)
    set(GGML_BUNDLE_CUDA_RUNTIME ${AICore_BUNDLE_CUDA_RUNTIME} CACHE BOOL
        "Internal: synced from AICore_BUNDLE_CUDA_RUNTIME" FORCE)
    set(GGML_CUDA_FORCE_MMQ ${AICore_CUDA_FORCE_MMQ} CACHE BOOL
        "Internal: synced from AICore_CUDA_FORCE_MMQ" FORCE)
    # SAM3D is part of the unified AICore task set. This private flag only
    # tells the fetched ggml build to wire its task-specific CUDA sources.
    set(GGML_USE_SAM3D_OPS ON CACHE BOOL
        "Internal: required by the unified AICore task set" FORCE)
    set(GGML_OPENCL_TARGET_VERSION ${AICore_OPENCL_TARGET_VERSION} CACHE STRING
        "Internal: synced from AICore_OPENCL_TARGET_VERSION" FORCE)
    set(GGML_CPU_ALL_VARIANTS ${AICore_CPU_ALL_VARIANTS} CACHE BOOL
        "Internal: synced from AICore_CPU_ALL_VARIANTS" FORCE)

    # Fixed ggml build policy for AICore (not user-facing CMake options).
    set(GGML_BUILD_SHARED ON CACHE BOOL
        "Internal: AICore requires dynamic ggml backend modules" FORCE)
    set(GGML_USE_LLAMAFILE ON CACHE BOOL
        "Internal: ggml tinyBLAS CPU matmul (llamafile)" FORCE)

    mark_as_advanced(
        GGML_USE_METAL GGML_USE_VULKAN GGML_USE_CUDA GGML_USE_SYCL GGML_SYCL_USE_DNN
        GGML_USE_OPENCL GGML_BUNDLE_CUDA_RUNTIME GGML_CUDA_FORCE_MMQ
        GGML_USE_SAM3D_OPS GGML_OPENCL_TARGET_VERSION
        GGML_CPU_ALL_VARIANTS GGML_BUILD_SHARED GGML_USE_LLAMAFILE)
endfunction()

function(_aicore_set_enabled_result aicore_var ggml_var help)
    unset(${aicore_var})
    if(DEFINED ${ggml_var})
        set(${aicore_var} ${${ggml_var}} CACHE BOOL "${help}" FORCE)
    else()
        set(${aicore_var} OFF CACHE BOOL "${help}" FORCE)
    endif()
endfunction()

function(aicore_sync_results_from_ggml)
    _aicore_set_enabled_result(AICore_METAL_ENABLED GGML_METAL_ENABLED
        "Read-only: AICore Metal backend built (from ggml configure)")
    _aicore_set_enabled_result(AICore_VULKAN_ENABLED GGML_VULKAN_ENABLED
        "Read-only: AICore Vulkan backend built (from ggml configure)")
    _aicore_set_enabled_result(AICore_CUDA_ENABLED GGML_CUDA_ENABLED
        "Read-only: AICore CUDA backend built (from ggml configure)")
    _aicore_set_enabled_result(AICore_SYCL_ENABLED GGML_SYCL_ENABLED
        "Read-only: AICore SYCL backend built (from ggml configure)")
    _aicore_set_enabled_result(AICore_OPENCL_ENABLED GGML_OPENCL_ENABLED
        "Read-only: AICore OpenCL backend built (from ggml configure)")
    mark_as_advanced(
        AICore_METAL_ENABLED AICore_VULKAN_ENABLED AICore_CUDA_ENABLED
        AICore_SYCL_ENABLED AICore_OPENCL_ENABLED)
endfunction()
