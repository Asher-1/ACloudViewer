# Backend discovery and compile options for the unique ggml source owner.
# Included by llama_runtime.cmake; paths are anchored at _LLAMA_RUNTIME_DIR.

# --- Backend selection ---
# ggml can compile multiple backends into one build (Vulkan + SYCL + CUDA + CPU).
# CPU is always built. GPU backends below are enabled when ON and dependencies are found.
#
# Platform defaults (user override: -DAICore_USE_*=ON/OFF via cmake/AICoreOptions.cmake):
#   Apple:     Metal ON, Vulkan/SYCL/OpenCL OFF
#   Linux/Win: Vulkan ON (auto-detect), SYCL/OpenCL OFF
if(APPLE)
    set(_GGML_METAL_DEFAULT ON)
    set(_GGML_VULKAN_DEFAULT OFF)
    set(_GGML_OPENCL_DEFAULT OFF)
elseif(WIN32 OR UNIX)
    set(_GGML_METAL_DEFAULT OFF)
    set(_GGML_VULKAN_DEFAULT ON)
    set(_GGML_OPENCL_DEFAULT OFF)
else()
    set(_GGML_METAL_DEFAULT OFF)
    set(_GGML_VULKAN_DEFAULT OFF)
    set(_GGML_OPENCL_DEFAULT OFF)
endif()

# CUDA backend (opt-in: NVIDIA toolchain required)
_ggml_read_synced_cache(GGML_USE_CUDA BOOL OFF
    "Internal: synced from AICore_USE_CUDA")
set(_GGML_CUDA_ENABLED OFF)
if(GGML_USE_CUDA)
    list(APPEND GGML_CMAKE_ARGS -DGGML_CUDA=ON)
    if(DEFINED CUDAToolkit_ROOT)
        list(APPEND GGML_CMAKE_ARGS -DCUDAToolkit_ROOT=${CUDAToolkit_ROOT})
    endif()

    # AICore CUDA is independent of the application's CUDA language targets.
    # When BUILD_CUDA_MODULE is OFF, the parent may leave the ExternalProject
    # compiler argument empty even though FindCUDAToolkit found a usable nvcc.
    if(NOT CMAKE_CUDA_COMPILER)
        if(CUDAToolkit_NVCC_EXECUTABLE)
            set(_ggml_cuda_compiler "${CUDAToolkit_NVCC_EXECUTABLE}")
        else()
            find_program(_ggml_cuda_compiler NAMES nvcc
                HINTS
                    "$ENV{CUDA_PATH}/bin"
                    "$ENV{CUDA_HOME}/bin"
                    "${CUDAToolkit_ROOT}/bin"
                    "/usr/local/cuda/bin")
        endif()
        if(NOT _ggml_cuda_compiler)
            message(FATAL_ERROR
                "AICore CUDA was requested, but no nvcc compiler was found. "
                "Set CMAKE_CUDA_COMPILER, CUDAToolkit_ROOT, CUDA_PATH, or CUDA_HOME.")
        endif()
        list(APPEND GGML_CMAKE_ARGS
            -DCMAKE_CUDA_COMPILER=${_ggml_cuda_compiler})
    endif()

    # Resolve from the dependency owner, including standalone validation builds.
    get_filename_component(_GGML_PROJECT_ROOT "${_LLAMA_RUNTIME_DIR}/../.." ABSOLUTE)
    include("${_GGML_PROJECT_ROOT}/cmake/CloudViewerMakeCudaArchitectures.cmake")
    cloudViewer_make_cuda_architectures(_ggml_cuda_archs)
    if(_ggml_cuda_archs)
        string(REPLACE ";" "$<SEMICOLON>" _ggml_cuda_archs_escaped "${_ggml_cuda_archs}")
        list(APPEND GGML_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=${_ggml_cuda_archs_escaped})
        message(STATUS "ggml: CUDA architectures = ${_ggml_cuda_archs}")
    else()
        list(APPEND GGML_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=native)
        message(STATUS "ggml: CUDA architectures = native (fallback)")
    endif()
    list(APPEND GGML_CMAKE_ARGS -DGGML_CUDA_NCCL=OFF)
    # FORCE_MMQ is synced from AICore_CUDA_FORCE_MMQ (default OFF = upstream
    # performance parity; ON = self-contained libggml-cuda.so without cuBLAS
    # DT_NEEDED for driver-only deployments). The historical companion MMQ
    # patch of the retired 0.21 chain is inert when GGML_CUDA_FORCE_MMQ=OFF.
    list(APPEND GGML_CMAKE_ARGS -DGGML_CUDA_FORCE_MMQ=${GGML_CUDA_FORCE_MMQ})
    # FAST_MATH is synced from AICore_CUDA_FAST_MATH (default ON = pristine
    # ggml `-use_fast_math`, the numerics every external cpp_ggml reference
    # except sam-3d-objects-ggml is built with; MapAnything CUDA bit-parity
    # with map-anything-ggml requires it). OFF mirrors the sam3d upstream
    # production build (see patches/sam3d_merged/0002 for the SiLU half of
    # that contract, which stays precise regardless of this flag). In the ON
    # build the sam3d_* nodes still get the OFF numerics through the
    # per-TU twins of patches/sam3d_merged/0005 (GGML_SAM3D_OPS=ON only).
    list(APPEND GGML_CMAKE_ARGS -DGGML_CUDA_FAST_MATH=${GGML_CUDA_FAST_MATH})
    # NOTE: the upstream sam-3d-objects-ggml production build replays CUDA
    # graphs (run_ggml.sh: GGML_CUDA_GRAPHS ON FORCE), but replay bypasses the
    # per-call host-side logic of our ported sam3d custom ops (spconv plan
    # selection and the masked-attention workspaces); the ss flow occupancy
    # collapses (58182 -> 4992 coords on the kidsroom fixture) because the
    # capture bakes the first call's internal scratch pointers into the
    # replay. patches/cuda_graphs/0001 now excludes graphs containing those
    # host-orchestrated custom ops from capture instead of disabling graphs
    # globally, so standard-op graphs (OCR det/rec, layout, facedetect, ...)
    # get capture/replay while sam3d keeps its per-call host logic.
    list(APPEND GGML_CMAKE_ARGS -DGGML_CUDA_GRAPHS=ON)
    set(_GGML_CUDA_ENABLED ON)
    message(STATUS "ggml: CUDA backend enabled (FORCE_MMQ=${GGML_CUDA_FORCE_MMQ}, FAST_MATH=${GGML_CUDA_FAST_MATH}, static cudart)")
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_CUDA=OFF)
endif()

# ── SAM 3D Objects operator set (patches/sam3d_merged) ──────────────────────
# SAM3D is part of the unified AICore task set. When CUDA is enabled the pinned
# PyTorch mem-eff attention headers and CUTLASS tarball are fetched here and
# handed to the ggml build through GGML_SAM3D_* directory caches; the vendored
# spconv sources live in 3rdparty/llama/sam3d/spconv. Without CUDA the op set
# still builds (CPU reference kernels + Vulkan scatter shader) and the ggml
# patch simply compiles the CUDA-only kernels out.
_ggml_read_synced_cache(GGML_USE_SAM3D_OPS BOOL ON
    "Internal: required by the unified AICore task set")
set(_GGML_SAM3D_DEPS_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/ggml/sam3d-deps")
if(GGML_USE_SAM3D_OPS AND _GGML_CUDA_ENABLED)
    # PyTorch 2.5.1 mem-efficient attention public header closure, pinned by
    # content hash (header-only; neither libtorch nor cuDNN is linked).
    set(_sam3d_memeff_dir "${_GGML_SAM3D_DEPS_DIR}/pytorch-v2.5.1-mem-eff-attention")
    set(_sam3d_pytorch_url "https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src")
    function(_ggml_sam3d_fetch_memeff_header relative_path expected_sha256)
        set(destination "${_sam3d_memeff_dir}/${relative_path}")
        if(EXISTS "${destination}")
            file(SHA256 "${destination}" actual_sha256)
            if(actual_sha256 STREQUAL expected_sha256)
                return()
            endif()
            file(REMOVE "${destination}")
        endif()
        get_filename_component(destination_dir "${destination}" DIRECTORY)
        file(MAKE_DIRECTORY "${destination_dir}")
        message(STATUS "ggml: fetching SAM3D pinned header ${relative_path}")
        file(DOWNLOAD "${_sam3d_pytorch_url}/${relative_path}" "${destination}"
            EXPECTED_HASH "SHA256=${expected_sha256}" TLS_VERIFY ON
            STATUS _sam3d_dl_status)
        list(GET _sam3d_dl_status 0 _sam3d_dl_code)
        if(NOT _sam3d_dl_code EQUAL 0)
            list(GET _sam3d_dl_status 1 _sam3d_dl_msg)
            message(FATAL_ERROR "Failed to fetch pinned PyTorch attention header ${relative_path}: ${_sam3d_dl_msg}")
        endif()
    endfunction()
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/debug_utils.h" "0bc7029c77ea42c8144e92814a7417c7b49acb4dee002fba0763a35a7ee27c02")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_pipelined.h" "220dcb8e76c6f95339f9ef8630aae000a2c65fc003f7063d0c1d2da7df5ef8d8")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_rescale_output.h" "01937cbd90f837ba0aa0674a111b56b7a2fdac93acd7bd46384066a26310f78c")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_thread_apply_logsumexp.h" "c5ca7711ae4f94cd2b1c3cd8275114b5740b804748ca4d08c501bed9f42f6941")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma.h" "5bdb4c6606d73a76fd740e762dcda90b175039d5e13cec18e1b074e1e9179397")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_base.h" "9281f07cebb654fb3107a4150a9b840a54a0318957ec8c4c9535586d6369b6c1")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_multistage.h" "2b73ea750e2f2455d42a2b6dcdf2553ed223a1efcd0865688866f3253dd80b41")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_pipelined.h" "71c2bb92d8a3ba5926cf27d1bf4657cd61b018f9cc6f03d937386bbb1f4876f1")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm/find_default_mma.h" "223613991c96520b6dbeeefeb160b948483f80cb60800e81f184c22a0d335994")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_accum_lambda_iterator.h" "ec666626f49dd4b6570a3f284735bf38bbbd5d56b53c49ffd3ff1f9345462784")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_from_smem.h" "aa18d0eb6812037d128471464a20887192c84193e8f78396b8d3b1a3442abce8")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h" "ce914b76b79d92c487a44cfc3ed19f41da947d0354672ab716489091fb0a7cbf")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/iterators/default_warp_iterator_from_smem.h" "c46d8c01507a4a049f784545379b323194750f9b34b3400a020f2ccc2bb912d8")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/iterators/epilogue_predicated_tile_iterator.h" "bee2b0cf715346a45c96a45a16bbe7f362a2a74d2ec5b1cd4410881a3920fd4f")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/iterators/make_residual_last.h" "17f4e1b6e4b473e21e106d2d4f71a07196a6849d41be8769c16fd2db43a0e666")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/iterators/predicated_tile_access_iterator_residual_last.h" "d17ec394229e52e693cea853694f0278f813624dc7cd497e2a2ace515d1d9ff6")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/iterators/predicated_tile_iterator_residual_last.h" "880a3383d716b50f5527564d01e066a8eabeace93a534562065c55fa4faaf220")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/iterators/transpose_warp_iterator.h" "8eb466e7e3013da32d0a083a3e5df978d7983b83a2dd2a5e362c526892ae894e")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/iterators/warp_iterator_from_smem.h" "3e97f7b3eae7cb5c21fb9b18761a6472b0b1b317923508c6a0cfcf8240983258")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h" "6019c38609117f0b7c28c7f2da4cbd24713aaa3d37b028d23da744bfd46dd2ec")
    _ggml_sam3d_fetch_memeff_header("ATen/native/transformers/cuda/mem_eff_attention/transform/tile_smem_loader.h" "798f55c4a47fadbaad11316d7a175f1063f63919069e034cf67debc946a2ecb0")

    # CUTLASS pinned tarball (BSD-3).
    set(_sam3d_cutlass_dir "${_GGML_SAM3D_DEPS_DIR}/cutlass-bbe579a9e3beb6ea6626d9227ec32d0dae119a49")
    if(NOT EXISTS "${_sam3d_cutlass_dir}/include/cutlass/cutlass.h")
        set(_sam3d_cutlass_tarball "${_GGML_SAM3D_DEPS_DIR}/cutlass-bbe579a9.tar.gz")
        message(STATUS "ggml: fetching SAM3D pinned CUTLASS tarball")
        file(DOWNLOAD
            "https://github.com/NVIDIA/cutlass/archive/bbe579a9e3beb6ea6626d9227ec32d0dae119a49.tar.gz"
            "${_sam3d_cutlass_tarball}"
            EXPECTED_HASH "SHA256=9fa1da6be3d2d9207b801d5768cbced59c202444a8c84b82325b0670f47f9d48"
            TLS_VERIFY ON
            STATUS _sam3d_cutlass_status)
        list(GET _sam3d_cutlass_status 0 _sam3d_cutlass_code)
        if(NOT _sam3d_cutlass_code EQUAL 0)
            list(GET _sam3d_cutlass_status 1 _sam3d_cutlass_msg)
            message(FATAL_ERROR "Failed to fetch pinned CUTLASS for SAM3D ops: ${_sam3d_cutlass_msg}")
        endif()
        file(ARCHIVE_EXTRACT INPUT "${_sam3d_cutlass_tarball}" DESTINATION "${_GGML_SAM3D_DEPS_DIR}")
        file(RENAME "${_GGML_SAM3D_DEPS_DIR}/cutlass-bbe579a9e3beb6ea6626d9227ec32d0dae119a49"
                    "${_sam3d_cutlass_dir}")
    endif()

    list(APPEND GGML_CMAKE_ARGS
        -DGGML_SAM3D_OPS=ON
        -DGGML_SAM3D_MEMEFF_DIR=${_sam3d_memeff_dir}
        -DGGML_SAM3D_MEMEFF_COMPAT_DIR=${_LLAMA_RUNTIME_DIR}/sam3d/pytorch-mem-eff-attention-compat
        -DGGML_SAM3D_CUTLASS_DIR=${_sam3d_cutlass_dir}
        -DGGML_SAM3D_SPCONV_DIR=${_LLAMA_RUNTIME_DIR}/sam3d/spconv)
    # GGML_CUDA_FAST_MATH is no longer forced OFF here: the sam3d upstream
    # production build resolves it OFF, but map-anything-ggml (pristine ggml)
    # resolves it ON. -use_fast_math is a per-translation-unit nvcc switch, so
    # patches/sam3d_merged/0005 compiles the kernels sam3d reaches a second
    # time without it (namespace ggml_cuda_precise) and routes sam3d_* nodes
    # there; both contracts hold in one libggml-cuda.so and sam3d bit-parity
    # evidence is valid in the default AICore_CUDA_FAST_MATH=ON build.
    message(STATUS "ggml: SAM3D operator set enabled (mem-eff + CUTLASS + spconv)")
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_SAM3D_OPS=${GGML_USE_SAM3D_OPS})
endif()

# Optional: redist CUDA runtime (cudart/cublas/cublasLt) into installer for
# driver-only deployment. OFF by default; never enabled in GitHub CI.
_ggml_read_synced_cache(GGML_BUNDLE_CUDA_RUNTIME BOOL OFF
    "Internal: synced from AICore_BUNDLE_CUDA_RUNTIME")
if(GGML_BUNDLE_CUDA_RUNTIME AND NOT GGML_USE_CUDA)
    message(FATAL_ERROR
        "AICore_BUNDLE_CUDA_RUNTIME=ON requires AICore_USE_CUDA=ON")
endif()
if(GGML_BUNDLE_CUDA_RUNTIME)
    message(STATUS "ggml: CUDA runtime bundling enabled (driver-only installer)")
endif()

# Metal backend (Apple; default ON on macOS/iOS)
_ggml_read_synced_cache(GGML_USE_METAL BOOL ${_GGML_METAL_DEFAULT}
    "Internal: synced from AICore_USE_METAL")
set(_GGML_METAL_ENABLED OFF)
if(GGML_USE_METAL)
    if(APPLE)
        list(APPEND GGML_CMAKE_ARGS -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON)
        set(_GGML_METAL_ENABLED ON)
        message(STATUS "ggml: Metal backend enabled")
    else()
        list(APPEND GGML_CMAKE_ARGS -DGGML_METAL=OFF)
        message(STATUS "ggml: Metal backend requested on non-Apple platform — skipping")
    endif()
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_METAL=OFF)
    if(APPLE)
        message(STATUS "ggml: Metal backend disabled (GGML_USE_METAL=OFF)")
    endif()
endif()

# Vulkan backend (portable release default: build-time Vulkan headers, glslc,
# and SPIR-V headers; end users still need a working driver/ICD).
# The llama candidate rejects missing glslc/Vulkan/SPIR-V dependencies.
# The old source owner retains its existing warning/disable behavior.
# GGML_VULKAN_ENABLED (below) is the read-only configure result.
_ggml_read_synced_cache(GGML_USE_VULKAN BOOL ${_GGML_VULKAN_DEFAULT}
    "Internal: synced from AICore_USE_VULKAN")
set(_GGML_VULKAN_ENABLED OFF)
if(GGML_USE_VULKAN)
    set(_GGML_VULKAN_HINTS)
    if(WIN32 AND DEFINED ENV{LOCALAPPDATA})
        set(_ACV_LOCAL_ROOT "$ENV{LOCALAPPDATA}/acloudviewer")
    elseif(DEFINED ENV{HOME})
        set(_ACV_LOCAL_ROOT "$ENV{HOME}/.local/share/acloudviewer")
    endif()
    if(DEFINED _ACV_LOCAL_ROOT)
        list(APPEND _GGML_VULKAN_HINTS
            "${_ACV_LOCAL_ROOT}/spirv-headers")
    endif()
    if(WIN32 AND NOT DEFINED ENV{VULKAN_SDK} AND DEFINED ENV{ProgramData})
        file(GLOB _ACV_WIN_VULKAN "$ENV{ProgramData}/ACloudViewer/VulkanSDK/*")
        list(SORT _ACV_WIN_VULKAN)
        list(REVERSE _ACV_WIN_VULKAN)
        foreach(_root IN LISTS _ACV_WIN_VULKAN)
            if(EXISTS "${_root}/Include/vulkan/vulkan_core.h")
                set(ENV{VULKAN_SDK} "${_root}")
                break()
            endif()
        endforeach()
    elseif(DEFINED ENV{HOME})
        if(NOT DEFINED ENV{VULKAN_SDK})
            file(GLOB _ACV_VULKAN_SETUPS
                "$ENV{HOME}/VulkanSDK/*/setup-env.sh"
                "$ENV{HOME}/VulkanSDK/*/*/setup-env.sh")
            list(SORT _ACV_VULKAN_SETUPS)
            list(REVERSE _ACV_VULKAN_SETUPS)
            foreach(_setup IN LISTS _ACV_VULKAN_SETUPS)
                get_filename_component(_sdk_root "${_setup}" DIRECTORY)
                if(EXISTS "${_sdk_root}/include/vulkan/vulkan_core.h")
                    set(ENV{VULKAN_SDK} "${_sdk_root}")
                    break()
                endif()
                if(EXISTS "${_sdk_root}/x86_64/include/vulkan/vulkan_core.h")
                    set(ENV{VULKAN_SDK} "${_sdk_root}/x86_64")
                    break()
                endif()
            endforeach()
        endif()
    endif()
    if(DEFINED ENV{VULKAN_SDK})
        list(APPEND CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK}")
        list(APPEND _GGML_VULKAN_HINTS
            "$ENV{VULKAN_SDK}"
            "$ENV{VULKAN_SDK}/include"
            "$ENV{VULKAN_SDK}/Include")
    endif()
    if(DEFINED ENV{ACLOUDVIEWER_SPIRV_HEADERS_DIR})
        get_filename_component(_ACV_SPIRV_PREFIX "$ENV{ACLOUDVIEWER_SPIRV_HEADERS_DIR}/../.." ABSOLUTE)
        list(APPEND CMAKE_PREFIX_PATH "${_ACV_SPIRV_PREFIX}")
        if(NOT SPIRV-Headers_DIR)
            set(SPIRV-Headers_DIR "$ENV{ACLOUDVIEWER_SPIRV_HEADERS_DIR}")
        endif()
    elseif(DEFINED _ACV_LOCAL_ROOT)
        set(_ACV_SPIRV_CMAKE "${_ACV_LOCAL_ROOT}/spirv-headers/share/cmake/SPIRV-Headers")
        if(EXISTS "${_ACV_SPIRV_CMAKE}/SPIRV-HeadersConfig.cmake" AND NOT SPIRV-Headers_DIR)
            set(SPIRV-Headers_DIR "${_ACV_SPIRV_CMAKE}")
            list(APPEND CMAKE_PREFIX_PATH "${_ACV_LOCAL_ROOT}/spirv-headers")
        endif()
    endif()
    if(DEFINED ENV{ACLOUDVIEWER_SPIRV_INCLUDE_DIR})
        list(APPEND _GGML_VULKAN_HINTS "$ENV{ACLOUDVIEWER_SPIRV_INCLUDE_DIR}")
    endif()

    # Older FindVulkan modules do not expose the glslc component. Locate the
    # loader/headers and shader compiler independently so Ubuntu 20.04 and the
    # official Windows/macOS SDKs follow the same contract.
    # Drop stale cache entries (removed SDK tree, switched glslc install path, etc.).
    if(Vulkan_INCLUDE_DIR AND NOT EXISTS "${Vulkan_INCLUDE_DIR}/vulkan/vulkan_core.h")
        unset(Vulkan_INCLUDE_DIR CACHE)
    endif()
    if(Vulkan_GLSLC_EXECUTABLE AND NOT EXISTS "${Vulkan_GLSLC_EXECUTABLE}")
        unset(Vulkan_GLSLC_EXECUTABLE CACHE)
    endif()
    if(Vulkan_LIBRARY AND NOT EXISTS "${Vulkan_LIBRARY}")
        unset(Vulkan_LIBRARY CACHE)
    endif()
    if(DEFINED ENV{ACLOUDVIEWER_VULKAN_LIBRARY} AND EXISTS "$ENV{ACLOUDVIEWER_VULKAN_LIBRARY}")
        set(Vulkan_LIBRARY "$ENV{ACLOUDVIEWER_VULKAN_LIBRARY}" CACHE FILEPATH "System Vulkan loader" FORCE)
    endif()
    find_package(Vulkan QUIET)
    if(DEFINED ENV{VULKAN_SDK} AND EXISTS "$ENV{VULKAN_SDK}/include/vulkan/vulkan_core.h")
        set(Vulkan_INCLUDE_DIR "$ENV{VULKAN_SDK}/include" CACHE PATH "Vulkan headers" FORCE)
    elseif(NOT Vulkan_INCLUDE_DIR OR NOT EXISTS "${Vulkan_INCLUDE_DIR}/vulkan/vulkan_core.h")
        find_path(Vulkan_INCLUDE_DIR NAMES vulkan/vulkan_core.h
            PATHS /usr/include ${CMAKE_SYSROOT}/usr/include
            NO_DEFAULT_PATH)
        if(Vulkan_INCLUDE_DIR)
            set(Vulkan_INCLUDE_DIR "${Vulkan_INCLUDE_DIR}" CACHE PATH "Vulkan headers" FORCE)
        endif()
    endif()
    if(DEFINED ENV{ACLOUDVIEWER_GLSLC} AND EXISTS "$ENV{ACLOUDVIEWER_GLSLC}")
        set(Vulkan_GLSLC_EXECUTABLE "$ENV{ACLOUDVIEWER_GLSLC}" CACHE FILEPATH "glslc from ACloudViewer env" FORCE)
    elseif(NOT Vulkan_GLSLC_EXECUTABLE OR NOT EXISTS "${Vulkan_GLSLC_EXECUTABLE}")
        unset(Vulkan_GLSLC_EXECUTABLE CACHE)
        find_program(Vulkan_GLSLC_EXECUTABLE NAMES glslc
            HINTS ${_GGML_VULKAN_HINTS} "$ENV{HOME}/.local/bin" /usr/local/bin
            PATH_SUFFIXES bin Bin)
    endif()

    find_package(SPIRV-Headers CONFIG QUIET)
    set(_GGML_SPIRV_HEADERS_FOUND ${SPIRV-Headers_FOUND})
    set(_GGML_SPIRV_INCLUDE_DIR)
    if(_GGML_SPIRV_HEADERS_FOUND)
        # Extract include dir from the imported target so we can forward it
        # to the ExternalProject (ggml-vulkan uses #include <spirv/unified1/spirv.hpp>
        # but doesn't link SPIRV-Headers::SPIRV-Headers).
        if(TARGET SPIRV-Headers::SPIRV-Headers)
            get_target_property(_GGML_SPIRV_INCLUDE_DIR
                SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)
        endif()
        if(NOT _GGML_SPIRV_INCLUDE_DIR AND SPIRV-Headers_DIR)
            get_filename_component(_GGML_SPIRV_INCLUDE_DIR
                "${SPIRV-Headers_DIR}/../../.." ABSOLUTE)
            set(_GGML_SPIRV_INCLUDE_DIR "${_GGML_SPIRV_INCLUDE_DIR}/include")
            if(NOT EXISTS "${_GGML_SPIRV_INCLUDE_DIR}/spirv/unified1/spirv.hpp")
                set(_GGML_SPIRV_INCLUDE_DIR)
            endif()
        endif()
    else()
        set(_GGML_VULKAN_INCLUDE_HINTS ${_GGML_VULKAN_HINTS})
        if(DEFINED ENV{ACLOUDVIEWER_SPIRV_INCLUDE_DIR})
            list(APPEND _GGML_VULKAN_INCLUDE_HINTS "$ENV{ACLOUDVIEWER_SPIRV_INCLUDE_DIR}")
        endif()
        if(Vulkan_INCLUDE_DIR)
            list(APPEND _GGML_VULKAN_INCLUDE_HINTS "${Vulkan_INCLUDE_DIR}")
        endif()
        if(Vulkan_INCLUDE_DIRS)
            list(APPEND _GGML_VULKAN_INCLUDE_HINTS ${Vulkan_INCLUDE_DIRS})
        endif()
        find_path(_GGML_SPIRV_INCLUDE_DIR
            NAMES spirv/unified1/spirv.hpp
            HINTS ${_GGML_VULKAN_INCLUDE_HINTS}
            PATH_SUFFIXES include Include)
        if(_GGML_SPIRV_INCLUDE_DIR)
            set(_GGML_SPIRV_HEADERS_FOUND ON)
        endif()
    endif()

    set(_GGML_VULKAN_HEADERS_OK OFF)
    if(Vulkan_INCLUDE_DIR AND EXISTS "${Vulkan_INCLUDE_DIR}/vulkan/vulkan_core.h")
        set(_GGML_VULKAN_HEADERS_OK ON)
    endif()

    if(Vulkan_FOUND AND _GGML_VULKAN_HEADERS_OK
            AND Vulkan_GLSLC_EXECUTABLE AND EXISTS "${Vulkan_GLSLC_EXECUTABLE}"
            AND _GGML_SPIRV_HEADERS_FOUND)
        list(APPEND GGML_CMAKE_ARGS -DGGML_VULKAN=ON)
        if(SPIRV-Headers_DIR)
            list(APPEND GGML_CMAKE_ARGS -DSPIRV-Headers_DIR=${SPIRV-Headers_DIR})
        endif()
        if(_GGML_SPIRV_INCLUDE_DIR)
            list(APPEND GGML_CMAKE_ARGS
                -DCMAKE_INCLUDE_PATH=${_GGML_SPIRV_INCLUDE_DIR}
                -DCMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES=${_GGML_SPIRV_INCLUDE_DIR}
                -DCMAKE_C_STANDARD_INCLUDE_DIRECTORIES=${_GGML_SPIRV_INCLUDE_DIR})
        endif()
        if(Vulkan_INCLUDE_DIR)
            list(APPEND GGML_CMAKE_ARGS -DVulkan_INCLUDE_DIR=${Vulkan_INCLUDE_DIR})
        endif()
        if(Vulkan_LIBRARY)
            list(APPEND GGML_CMAKE_ARGS -DVulkan_LIBRARY=${Vulkan_LIBRARY})
        endif()
        if(Vulkan_GLSLC_EXECUTABLE)
            list(APPEND GGML_CMAKE_ARGS -DVulkan_GLSLC_EXECUTABLE=${Vulkan_GLSLC_EXECUTABLE})
        endif()
        set(_GGML_VULKAN_ENABLED ON)
        message(STATUS "ggml: Vulkan backend enabled "
                       "(glslc=${Vulkan_GLSLC_EXECUTABLE}, SPIR-V headers=${_GGML_SPIRV_INCLUDE_DIR}${SPIRV-Headers_DIR})")
    else()
        list(APPEND GGML_CMAKE_ARGS -DGGML_VULKAN=OFF)
        set(_GGML_VULKAN_MISSING)
        if(NOT Vulkan_FOUND)
            list(APPEND _GGML_VULKAN_MISSING "Vulkan loader")
        endif()
        if(NOT _GGML_VULKAN_HEADERS_OK)
            if(Vulkan_INCLUDE_DIR)
                list(APPEND _GGML_VULKAN_MISSING "Vulkan headers (${Vulkan_INCLUDE_DIR})")
            else()
                list(APPEND _GGML_VULKAN_MISSING "Vulkan headers")
            endif()
        endif()
        if(NOT Vulkan_GLSLC_EXECUTABLE OR NOT EXISTS "${Vulkan_GLSLC_EXECUTABLE}")
            list(APPEND _GGML_VULKAN_MISSING "glslc")
        endif()
        if(NOT _GGML_SPIRV_HEADERS_FOUND)
            list(APPEND _GGML_VULKAN_MISSING "spirv/unified1/spirv.hpp")
        endif()
        string(JOIN ", " _GGML_VULKAN_MISSING_TEXT ${_GGML_VULKAN_MISSING})
        if(WIN32)
            set(_GGML_VULKAN_SETUP_HINT
                "Run: .\\util\\vulkan\\install_vulkan_sdk_windows.ps1 (or . \$env:LOCALAPPDATA\\acloudviewer\\acloudviewer-vulkan-env.ps1)")
        elseif(APPLE)
            set(_GGML_VULKAN_SETUP_HINT
                "Vulkan is not supported on macOS; use Metal (-DAICore_USE_VULKAN=OFF)")
        else()
            set(_GGML_VULKAN_SETUP_HINT
                "Run: util/install_deps_ubuntu.sh assume-yes or util/vulkan/install_vulkan_env.sh")
        endif()
        # The candidate must build the backend selected by its migration
        # manifest. A successful CPU-only configure cannot validate Vulkan.
        if(_GGML_OWNER_TARGET STREQUAL "ext_llama")
            message(FATAL_ERROR
                "llama.cpp: requested Vulkan backend dependencies are missing: "
                "${_GGML_VULKAN_MISSING_TEXT}. ${_GGML_VULKAN_SETUP_HINT}")
        endif()
        message(WARNING
            "AICore_USE_VULKAN=ON but Vulkan dependencies are missing: "
            "${_GGML_VULKAN_MISSING_TEXT}. Vulkan backend disabled. "
            "${_GGML_VULKAN_SETUP_HINT}")
        list(APPEND GGML_CMAKE_ARGS -DGGML_VULKAN=OFF)
        set(_GGML_VULKAN_ENABLED OFF)
    endif()
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_VULKAN=OFF)
endif()

# SYCL backend (Intel GPU). Keep this explicitly opt-in until release packages
# carry and validate the matching oneAPI runtime; upstream does not promise a
# standalone binary that works without it.
_ggml_read_synced_cache(GGML_USE_SYCL BOOL OFF
    "Internal: synced from AICore_USE_SYCL")
_ggml_read_synced_cache(GGML_SYCL_USE_DNN BOOL ON
    "Internal: synced from AICore_SYCL_USE_DNN")
set(_GGML_SYCL_ENABLED OFF)
if(GGML_USE_SYCL)
    if(APPLE)
        message(FATAL_ERROR "AICore SYCL is not supported on macOS; use Metal")
    elseif(NOT GGML_BUILD_SHARED)
        message(FATAL_ERROR
            "AICore_USE_SYCL=ON requires dynamic ggml backend modules")
    else()
        find_program(_GGML_SYCL_CXX_COMPILER NAMES icpx dpcpp)
        find_program(_GGML_SYCL_C_COMPILER NAMES icx)
        if(_GGML_SYCL_CXX_COMPILER AND _GGML_SYCL_C_COMPILER)
            list(APPEND GGML_CMAKE_ARGS
                -DCMAKE_C_COMPILER=${_GGML_SYCL_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${_GGML_SYCL_CXX_COMPILER}
                -DGGML_SYCL=ON
                -DGGML_SYCL_TARGET=INTEL
                -DGGML_SYCL_DNN=${GGML_SYCL_USE_DNN}
                -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON)
            set(_GGML_SYCL_ENABLED ON)
            message(STATUS "ggml: SYCL backend enabled (${_GGML_SYCL_CXX_COMPILER})")
        else()
            list(APPEND GGML_CMAKE_ARGS -DGGML_SYCL=OFF)
            message(STATUS "ggml: oneAPI compiler not found - skipping SYCL backend")
        endif()
    endif()
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_SYCL=OFF)
endif()

# OpenCL is retained only as an explicit Adreno/legacy developer backend.
# Upstream's desktop operation coverage is substantially below SYCL/Vulkan.
_ggml_read_synced_cache(GGML_USE_OPENCL BOOL ${_GGML_OPENCL_DEFAULT}
    "Internal: synced from AICore_USE_OPENCL")
_ggml_read_synced_cache(GGML_OPENCL_TARGET_VERSION STRING "200"
    "Internal: synced from AICore_OPENCL_TARGET_VERSION")
set_property(CACHE GGML_OPENCL_TARGET_VERSION PROPERTY STRINGS 120 200 300)
if(NOT GGML_OPENCL_TARGET_VERSION MATCHES "^(120|200|300)$")
    message(FATAL_ERROR
        "GGML_OPENCL_TARGET_VERSION must be one of 120, 200, or 300")
endif()
set(_GGML_OPENCL_ENABLED OFF)
if(GGML_USE_OPENCL)
    if(APPLE)
        list(APPEND GGML_CMAKE_ARGS -DGGML_OPENCL=OFF)
        message(STATUS "ggml: OpenCL disabled on macOS (system OpenCL is deprecated and unusable for ggml; use Metal)")
    else()
        find_package(OpenCL QUIET)
        find_package(Python3 QUIET COMPONENTS Interpreter)
        set(_GGML_OPENCL_HEADERS_OK OFF)
        if(GGML_OPENCL_TARGET_VERSION STREQUAL "300")
            set(_GGML_OPENCL_REQUIRED_MACRO "CL_VERSION_3_0")
        elseif(GGML_OPENCL_TARGET_VERSION STREQUAL "200")
            set(_GGML_OPENCL_REQUIRED_MACRO "CL_VERSION_2_0")
        else()
            set(_GGML_OPENCL_REQUIRED_MACRO "CL_VERSION_1_2")
        endif()
        if(OpenCL_FOUND)
            set(_ocl_include_dir "")
            if(OpenCL_INCLUDE_DIRS)
                list(GET OpenCL_INCLUDE_DIRS 0 _ocl_include_dir)
            elseif(OpenCL_INCLUDE_DIR)
                set(_ocl_include_dir "${OpenCL_INCLUDE_DIR}")
            endif()
            if(_ocl_include_dir AND EXISTS "${_ocl_include_dir}/CL/cl.h")
                if(EXISTS "${_ocl_include_dir}/CL/cl_version.h")
                    file(READ "${_ocl_include_dir}/CL/cl_version.h" _ggml_opencl_ver_h LIMIT 8192)
                endif()
                file(READ "${_ocl_include_dir}/CL/cl.h" _ggml_opencl_cl_h LIMIT 16384)
                if(_ggml_opencl_cl_h MATCHES "#define ${_GGML_OPENCL_REQUIRED_MACRO}"
                   OR _ggml_opencl_ver_h MATCHES "#define ${_GGML_OPENCL_REQUIRED_MACRO}")
                    set(_GGML_OPENCL_HEADERS_OK ON)
                endif()
            endif()
        endif()
        if(OpenCL_FOUND AND Python3_FOUND AND _GGML_OPENCL_HEADERS_OK)
            list(APPEND GGML_CMAKE_ARGS
                -DGGML_OPENCL=ON
                -DGGML_OPENCL_TARGET_VERSION=${GGML_OPENCL_TARGET_VERSION})
            set(_GGML_OPENCL_ENABLED ON)
            message(STATUS "ggml: OpenCL backend enabled (API target ${GGML_OPENCL_TARGET_VERSION})")
            # The dependency embeds OpenCL kernels at configure time; forward a real
            # interpreter path (pyenv shims are rejected by FindPython3).
            if(Python3_EXECUTABLE)
                get_filename_component(_ggml_opencl_python "${Python3_EXECUTABLE}" REALPATH)
            else()
                find_program(_ggml_opencl_python
                    NAMES python3 python
                    NO_PYENV_SHIM)
            endif()
            if(_ggml_opencl_python)
                list(APPEND GGML_CMAKE_ARGS
                    -DPython3_EXECUTABLE=${_ggml_opencl_python})
            endif()
        else()
            list(APPEND GGML_CMAKE_ARGS -DGGML_OPENCL=OFF)
            if(NOT OpenCL_FOUND)
                message(STATUS "ggml: OpenCL not found — skipping OpenCL backend "
                                "(install ocl-icd-opencl-dev / OpenCL ICD)")
            elseif(NOT _GGML_OPENCL_HEADERS_OK)
                message(STATUS "ggml: ${_GGML_OPENCL_REQUIRED_MACRO} headers not found — skipping OpenCL backend "
                                "(install Khronos OpenCL-Headers or lower GGML_OPENCL_TARGET_VERSION)")
            endif()
            if(NOT Python3_FOUND)
                message(STATUS "ggml: Python3 not found — skipping OpenCL backend "
                                "(required to embed OpenCL kernels at build time)")
            endif()
        endif()
    endif()
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_OPENCL=OFF)
    message(STATUS "ggml: OpenCL backend disabled (GGML_USE_OPENCL=OFF)")
endif()

# The standalone ggml BLAS backend is deliberately disabled. It implements only
# a subset of graph operations, so DA3/FreeSplatter require a mixed BLAS + CPU
# scheduler that is slower than ggml's optimized CPU backend and has caused
# allocator failures. CPU inference uses only ggml-cpu (including its portable
# llamafile kernels); GPU acceleration is provided by Vulkan/Metal.
set(GGML_USE_BLAS OFF CACHE BOOL
    "Standalone ggml BLAS backend is unsupported by AICore" FORCE)
set(_GGML_EXTERNAL_DEPENDS)
list(APPEND GGML_CMAKE_ARGS -DGGML_BLAS=OFF)
if(APPLE)
    # Accelerate is a system CPU framework used by ggml-cpu, not the separate
    # partial-graph ggml-blas backend.
    list(APPEND GGML_CMAKE_ARGS -DGGML_ACCELERATE=ON)
endif()

# tinyBLAS (llamafile) for optimized CPU matmul — fixed ON for AICore builds.
_ggml_read_synced_cache(GGML_USE_LLAMAFILE BOOL ON
    "Internal: ggml tinyBLAS CPU matmul (llamafile)")
list(APPEND GGML_CMAKE_ARGS -DGGML_LLAMAFILE=${GGML_USE_LLAMAFILE})

# CPU micro-architecture variants (llama.cpp release style).
_ggml_read_synced_cache(GGML_CPU_ALL_VARIANTS BOOL OFF
    "Internal: synced from AICore_CPU_ALL_VARIANTS")
list(APPEND GGML_CMAKE_ARGS -DGGML_CPU_ALL_VARIANTS=${GGML_CPU_ALL_VARIANTS})

# CPU optimizations
# GGML v0.17+ forbids GGML_NATIVE with GGML_BACKEND_DL (shared/dynamic mode).
# Single ggml-cpu: runtime feature detection in one module.
# GGML_CPU_ALL_VARIANTS: per-ISA modules selected at runtime by cpu-feats.
if(GGML_CPU_ALL_VARIANTS)
    list(APPEND GGML_CMAKE_ARGS -DGGML_NATIVE=OFF)
elseif(GGML_BUILD_SHARED)
    list(APPEND GGML_CMAKE_ARGS -DGGML_NATIVE=OFF)
elseif(NOT CMAKE_CROSSCOMPILING)
    list(APPEND GGML_CMAKE_ARGS -DGGML_NATIVE=ON)
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_NATIVE=OFF)
endif()

# OpenMP
# On macOS the ggml-cpu dynamic module links its own libomp.dylib.  When the
# host process (ACloudViewer / Python) already loaded a copy, the duplicate
# triggers "OMP: Error #15 … already initialized" and aborts.  Disable ggml's
# own OpenMP on Apple; Accelerate + Metal cover the workload.
if(APPLE)
    list(APPEND GGML_CMAKE_ARGS -DGGML_OPENMP=OFF)
elseif(WITH_OPENMP)
    list(APPEND GGML_CMAKE_ARGS -DGGML_OPENMP=ON)
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_OPENMP=OFF)
endif()

