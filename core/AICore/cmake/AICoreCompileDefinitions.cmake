# Apply AICore compile-time configuration macros to a target.
#
# These mirror CMake AICore_* backend options for #if branches in src/ only.
# They are NOT a second public API — use aicore_device_* at runtime when possible.
#
#   AICORE_BACKEND_DL  — ggml backends are loadable modules (default packaging)
#   AICORE_CUDA_STATIC_LINKED — CUDA linked into libAICore (non-DL developer builds)
#   AICORE_AUTO_INCLUDE_CUDA   — CUDA in Auto fallback (when AICore_CUDA_ENABLED)
#   AICORE_CUDA_ALIKED        — ALIKED CUDA custom kernels (when AICore_CUDA_ENABLED)
#   AICORE_VULKAN_ALIKED       — ggml-vulkan-aliked custom compute (when AICore_VULKAN_ENABLED)

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
        target_compile_definitions(${target} PRIVATE AICORE_AUTO_INCLUDE_CUDA)
        # core/AICore/src/aliked/aliked_cuda.cu custom DCN/DKD/SDDH kernels
        target_compile_definitions(${target} PRIVATE AICORE_CUDA_ALIKED)
    endif()
    if(AICore_VULKAN_ENABLED)
        # 3rdparty/ggml/patches/aliked/0001-vulkan-aliked-custom-compute.patch
        target_compile_definitions(${target} PRIVATE AICORE_VULKAN_ALIKED)
    endif()
endfunction()
