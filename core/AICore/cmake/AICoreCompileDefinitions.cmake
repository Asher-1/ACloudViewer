# Apply AICore compile-time configuration macros to a target.
#
# MACRO SEMANTIC BOUNDARY (single source of truth):
#   These macros answer exactly ONE question — "what did the build produce?"
#   (link mode, which source files were compiled). They must NEVER decide
#   runtime behavior such as per-backend data flows (f16 casts, q8 direct
#   conv): the actual device is only known after the BackendLease resolves
#   at runtime, so runtime capability checks go through the aicore_device_* C
#   API or backend-name detection (see BackendCtx::is_cuda/is_vulkan in
#   src/tasks/yolo/backend.hpp and the rmbg GraphBuilder for the pattern).
#
# Complete macro list (do not add task-specific switches here):
#   AICORE_BACKEND_DL       — ggml backends are loadable modules (default packaging)
#   AICORE_CUDA_STATIC_LINKED — CUDA linked into libAICore (non-DL developer builds)
#   AICORE_CUDA_BUILT       — a CUDA backend was built; orders Auto device priority
#   AICORE_CUDA_ALIKED      — ALIKED CUDA custom kernels (aliked_cuda.cu)
#   AICORE_VULKAN_ALIKED    — ALIKED Vulkan dispatch sources + ggml-vulkan-aliked patch
#   AICore_HAS_CVLOG        — CVLog integration (set in core/AICore/CMakeLists.txt)
#
# There is intentionally NO AICORE_VULKAN_BUILT: auto_backend_ids() in
# src/common/ggml_backend_utils.hpp already defaults to Vulkan on Linux/Windows
# when no CUDA backend was built, so Vulkan needs no compile-time priority
# macro (CUDA needs AICORE_CUDA_BUILT only because it must be ordered BEFORE
# Vulkan). Runtime device checks go through the aicore_device_* C API or
# ggml_backend_is_* detection (see BackendCtx::is_cuda/is_vulkan in
# src/tasks/yolo/backend.hpp).

function(aicore_apply_compile_definitions target)
    if(NOT DEFINED AICore_VULKAN_ENABLED)
        set(AICore_VULKAN_ENABLED OFF)
    endif()
    if(GGML_DYNAMIC_BACKENDS)
        target_compile_definitions(${target} PRIVATE AICORE_BACKEND_DL)
    elseif(AICore_CUDA_ENABLED)
        target_compile_definitions(${target} PRIVATE AICORE_CUDA_STATIC_LINKED)
    endif()
    if(AICore_CUDA_ENABLED)
        # A CUDA backend exists in this build: Auto resolution prefers it
        # (CUDA -> Vulkan -> CPU on Linux/Windows).
        target_compile_definitions(${target} PRIVATE AICORE_CUDA_BUILT)
        # core/AICore/src/tasks/aliked/aliked_cuda.cu custom DCN/DKD/SDDH kernels
        target_compile_definitions(${target} PRIVATE AICORE_CUDA_ALIKED)
    endif()
    if(AICore_VULKAN_ENABLED)
        # 3rdparty/ggml/patches/aliked/0001-vulkan-aliked-custom-compute.patch
        # (ALIKED-only: compiles the vulkan dispatch sources and enables the
        # patched ggml-vulkan-aliked custom compute ops.)
        target_compile_definitions(${target} PRIVATE AICORE_VULKAN_ALIKED)
    endif()
endfunction()
