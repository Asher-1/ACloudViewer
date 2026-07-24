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
    "Redist CUDA runtime libs into installer for driver-only targets (requires AICore_USE_CUDA=ON)"
    OFF)
set(AICore_OPENCL_TARGET_VERSION "200" CACHE STRING
    "OpenCL host API target for AICore (120, 200, or 300)")
set_property(CACHE AICore_OPENCL_TARGET_VERSION PROPERTY STRINGS 120 200 300)
option(AICore_CPU_ALL_VARIANTS
    "Build all ggml CPU micro-architecture backend variants (llama.cpp release style; +~15MB on x86)"
    OFF)

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
        GGML_USE_OPENCL GGML_BUNDLE_CUDA_RUNTIME GGML_OPENCL_TARGET_VERSION
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
