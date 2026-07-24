# Exports: ${GGML_INCLUDE_DIRS}
# Exports: ${GGML_LIB_DIR}
# Exports: ${GGML_LIBRARIES}

include(ExternalProject)

# GGML_ENABLED and backend toggles are owned by cmake/AICoreOptions.cmake (AICore_*).
# aicore_sync_options_to_ggml() must run before including this file.
macro(_ggml_read_synced_cache var type default help)
    if(NOT DEFINED CACHE{${var}})
        set(${var} ${default} CACHE ${type} "${help}" FORCE)
        mark_as_advanced(${var})
    endif()
endmacro()

_ggml_read_synced_cache(GGML_ENABLED BOOL ON
    "Internal: synced from AICore_ENABLED")

if(NOT GGML_ENABLED)
    message(STATUS "ggml is disabled")
    return()
endif()

set(GGML_VERSION "0.17.0")
set(GGML_URL "https://github.com/ggml-org/ggml/archive/refs/tags/v${GGML_VERSION}.tar.gz")
set(GGML_SHA256 "49ed958226dd75ea13b3b493150181e3a3ca7dc28c20a3d1f00d23e94cbf7a47")

# Dynamic backend mode: accelerator backends are built as separate shared libraries
# loaded at runtime (GGML_BACKEND_DL). Fixed ON for AICore — see AICoreOptions sync.
_ggml_read_synced_cache(GGML_BUILD_SHARED BOOL ON
    "Internal: AICore requires dynamic ggml backend modules")

# Reuse project-wide ExternalProject_CMAKE_ARGS_hidden (includes compiler,
# build type, MSVC runtime, PIC, visibility, CUDA compiler, etc.)
set(GGML_CMAKE_ARGS
    ${ExternalProject_CMAKE_ARGS_hidden}
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DGGML_BUILD_TESTS=OFF
    -DGGML_BUILD_EXAMPLES=OFF
)

if(GGML_BUILD_SHARED)
    list(APPEND GGML_CMAKE_ARGS -DBUILD_SHARED_LIBS=ON -DGGML_STATIC=OFF)
    list(APPEND GGML_CMAKE_ARGS -DGGML_BACKEND_DL=ON)
    if(WIN32)
        # MSVC links import libraries from lib/, while the corresponding
        # runtime DLLs are installed to bin/.
        set(_GGML_LIB_PREFIX ${CMAKE_IMPORT_LIBRARY_PREFIX})
        set(_GGML_LIB_SUFFIX ${CMAKE_IMPORT_LIBRARY_SUFFIX})
        set(_GGML_RUNTIME_SUBDIR bin)
    else()
        set(_GGML_LIB_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
        set(_GGML_LIB_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(_GGML_RUNTIME_SUBDIR ${CloudViewer_INSTALL_LIB_DIR})
    endif()
else()
    list(APPEND GGML_CMAKE_ARGS -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON)
    set(_GGML_LIB_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
    set(_GGML_LIB_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(_GGML_RUNTIME_SUBDIR ${CloudViewer_INSTALL_LIB_DIR})
endif()

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

    include(${CMAKE_SOURCE_DIR}/cmake/CloudViewerMakeCudaArchitectures.cmake)
    cloudViewer_make_cuda_architectures(_ggml_cuda_archs)
    if(_ggml_cuda_archs)
        string(REPLACE ";" "$<SEMICOLON>" _ggml_cuda_archs_escaped "${_ggml_cuda_archs}")
        list(APPEND GGML_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=${_ggml_cuda_archs_escaped})
        message(STATUS "ggml: CUDA architectures = ${_ggml_cuda_archs}")
    else()
        list(APPEND GGML_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=native)
        message(STATUS "ggml: CUDA architectures = native (fallback)")
    endif()
    list(APPEND GGML_CMAKE_ARGS -DGGML_CUDA_F16=ON)
    list(APPEND GGML_CMAKE_ARGS -DGGML_CUDA_NCCL=OFF)
    set(_GGML_CUDA_ENABLED ON)
    message(STATUS "ggml: CUDA backend enabled")
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_CUDA=OFF)
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
# GGML_USE_VULKAN=ON requires glslc/Vulkan/SPIR-V deps (configure fails if missing).
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
    if(NOT _GGML_SPIRV_HEADERS_FOUND)
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
                -DCMAKE_INCLUDE_PATH=${_GGML_SPIRV_INCLUDE_DIR})
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
                "Run: .\\util\\install_vulkan_sdk_windows.ps1 (or . \$env:LOCALAPPDATA\\acloudviewer\\acloudviewer-vulkan-env.ps1)")
        elseif(APPLE)
            set(_GGML_VULKAN_SETUP_HINT
                "Run: util/install_vulkan_env.sh (macOS) or util/install_vulkan_sdk_macos.sh")
        else()
            set(_GGML_VULKAN_SETUP_HINT
                "Run: util/install_deps_ubuntu.sh assume-yes or util/install_vulkan_env.sh")
        endif()
        message(FATAL_ERROR
            "AICore_USE_VULKAN=ON but Vulkan dependencies are missing: "
            "${_GGML_VULKAN_MISSING_TEXT}. ${_GGML_VULKAN_SETUP_HINT}")
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
            # ext_ggml embeds OpenCL kernels at configure time; forward a real
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
if(WITH_OPENMP)
    list(APPEND GGML_CMAKE_ARGS -DGGML_OPENMP=ON)
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_OPENMP=OFF)
endif()

# --- Build byproducts ---
# Link artifacts are import libraries on Windows shared builds, otherwise the
# shared/static libraries themselves.
set(GGML_BUILD_BYPRODUCTS
    <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_GGML_LIB_PREFIX}ggml${_GGML_LIB_SUFFIX}
    <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-base${_GGML_LIB_SUFFIX}
)
if(GGML_BUILD_SHARED)
    # GGML_BACKEND_DL mode: all backends (including ggml-cpu) are MODULE
    # libraries installed to bin/ with .so extension (even on macOS).
    set(_GGML_MODULE_SUFFIX ".so")
    if(WIN32)
        set(_GGML_MODULE_SUFFIX ".dll")
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${CMAKE_SHARED_LIBRARY_PREFIX}ggml${CMAKE_SHARED_LIBRARY_SUFFIX}
            <INSTALL_DIR>/bin/${CMAKE_SHARED_LIBRARY_PREFIX}ggml-base${CMAKE_SHARED_LIBRARY_SUFFIX}
        )
    endif()
    if(GGML_CPU_ALL_VARIANTS)
        # Baseline CPU module always built; extra ISA variants are discovered at
        # packaging time via libggml-cpu-* glob (compiler-adaptive patch).
        if(APPLE)
            set(_GGML_CPU_BYPRODUCT ggml-cpu-apple_m1)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|ARM64|arm64)$")
            set(_GGML_CPU_BYPRODUCT ggml-cpu-armv8.0_1)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
            set(_GGML_CPU_BYPRODUCT ggml-cpu-x64)
        else()
            set(_GGML_CPU_BYPRODUCT ggml-cpu)
        endif()
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${_GGML_LIB_PREFIX}${_GGML_CPU_BYPRODUCT}${_GGML_MODULE_SUFFIX})
    else()
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${_GGML_LIB_PREFIX}ggml-cpu${_GGML_MODULE_SUFFIX})
    endif()
    if(_GGML_CUDA_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${_GGML_LIB_PREFIX}ggml-cuda${_GGML_MODULE_SUFFIX})
    endif()
    if(_GGML_METAL_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${_GGML_LIB_PREFIX}ggml-metal${_GGML_MODULE_SUFFIX})
    endif()
    if(_GGML_VULKAN_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${_GGML_LIB_PREFIX}ggml-vulkan${_GGML_MODULE_SUFFIX})
    endif()
    if(_GGML_OPENCL_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${_GGML_LIB_PREFIX}ggml-opencl${_GGML_MODULE_SUFFIX})
    endif()
    if(_GGML_SYCL_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${_GGML_LIB_PREFIX}ggml-sycl${_GGML_MODULE_SUFFIX})
    endif()
else()
    # Static mode: all backends are regular static libs in lib/.
    list(APPEND GGML_BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-cpu${_GGML_LIB_SUFFIX}
    )
    if(_GGML_CUDA_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-cuda${_GGML_LIB_SUFFIX})
    endif()
    if(_GGML_METAL_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-metal${_GGML_LIB_SUFFIX})
    endif()
    if(_GGML_VULKAN_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-vulkan${_GGML_LIB_SUFFIX})
    endif()
    if(_GGML_OPENCL_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-opencl${_GGML_LIB_SUFFIX})
    endif()
endif()

set(_GGML_PATCH_PARTS)
if(GGML_CPU_ALL_VARIANTS)
    list(APPEND _GGML_PATCH_PARTS
        "${CMAKE_COMMAND} -E env python3 \"${CMAKE_CURRENT_LIST_DIR}/patches/apply_cpu_all_variants_compiler_checks.py\" \"<SOURCE_DIR>\"")
endif()
if(APPLE AND _GGML_METAL_ENABLED)
    list(APPEND _GGML_PATCH_PARTS
        "${CMAKE_COMMAND} -E env python3 \"${CMAKE_CURRENT_LIST_DIR}/patches/apply_metal_conv_transpose_opt.py\" \"<SOURCE_DIR>\"")
endif()
if(_GGML_PATCH_PARTS)
    list(JOIN _GGML_PATCH_PARTS " && " _GGML_PATCH_SHELL)
    if(WIN32)
        set(_GGML_PATCH_COMMAND cmd /c "${_GGML_PATCH_SHELL}")
    else()
        set(_GGML_PATCH_COMMAND bash -c "${_GGML_PATCH_SHELL}")
    endif()
else()
    set(_GGML_PATCH_COMMAND "")
endif()

set(_GGML_EXTERNAL_DEPENDS_ARGS)
if(_GGML_EXTERNAL_DEPENDS)
    set(_GGML_EXTERNAL_DEPENDS_ARGS DEPENDS ${_GGML_EXTERNAL_DEPENDS})
endif()

ExternalProject_Add(ext_ggml
    PREFIX ggml
    INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
    URL ${GGML_URL}
    URL_HASH SHA256=${GGML_SHA256}
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/ggml"
    UPDATE_COMMAND ""
    PATCH_COMMAND ${_GGML_PATCH_COMMAND}
    CMAKE_ARGS ${GGML_CMAKE_ARGS}
    BUILD_BYPRODUCTS ${GGML_BUILD_BYPRODUCTS}
    ${_GGML_EXTERNAL_DEPENDS_ARGS}
)

set(GGML_INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR})
set(GGML_INCLUDE_DIRS ${GGML_INSTALL_DIR}/include)
set(GGML_LIB_DIR ${GGML_INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR})
set(GGML_RUNTIME_LIB_DIR ${GGML_INSTALL_DIR}/${_GGML_RUNTIME_SUBDIR})
set(GGML_MODULE_DIR ${GGML_INSTALL_DIR}/bin)

# --- Create imported interface target ---
add_library(3rdparty_ggml INTERFACE)
add_dependencies(3rdparty_ggml ext_ggml)

target_include_directories(3rdparty_ggml INTERFACE ${GGML_INCLUDE_DIRS})

# In SHARED + GGML_BACKEND_DL mode, ggml-cpu is a MODULE library loaded at
# runtime by ggml_backend_load_all_from_path().  We only link the core shared
# libs (ggml + ggml-base).  In STATIC mode, ggml-cpu is a regular static lib
# and must be linked explicitly.
if(GGML_BUILD_SHARED)
    target_link_libraries(3rdparty_ggml INTERFACE
        ${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml${_GGML_LIB_SUFFIX}
        ${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-base${_GGML_LIB_SUFFIX}
    )
else()
    target_link_libraries(3rdparty_ggml INTERFACE
        ${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml${_GGML_LIB_SUFFIX}
        ${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-cpu${_GGML_LIB_SUFFIX}
        ${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-base${_GGML_LIB_SUFFIX}
    )
endif()

# --- GPU backend linking ---
# In SHARED mode, GPU backends are separate .so/.dylib files loaded at runtime
# via ggml_backend_load_all(). They are NOT linked into 3rdparty_ggml, so
# libAICore.so has no hard GPU runtime dependencies (no DT_NEEDED for
# libcuda.so.1, etc.). This is what enables the Python wheel to be imported
# on machines without GPU drivers.
#
# In explicit STATIC mode, backends are linked into AICore and all GPU symbols
# must resolve at process load time. Release app and wheel builds use dynamic
# backends so optional accelerators can fail without breaking CPU inference.

if(NOT GGML_BUILD_SHARED)
    # CUDA: reuse CUDAToolkit already found by the parent project
    if(_GGML_CUDA_ENABLED)
        target_link_libraries(3rdparty_ggml INTERFACE
            ${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-cuda${_GGML_LIB_SUFFIX}
            CUDA::cuda_driver
            CUDA::cudart
        )
        if(TARGET CUDA::cublas)
            target_link_libraries(3rdparty_ggml INTERFACE CUDA::cublas)
        endif()
        if(TARGET CUDA::cublasLt)
            target_link_libraries(3rdparty_ggml INTERFACE CUDA::cublasLt)
        endif()
    endif()

    # Metal (macOS/iOS)
    if(_GGML_METAL_ENABLED)
        target_link_libraries(3rdparty_ggml INTERFACE
            ${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-metal${_GGML_LIB_SUFFIX}
            "-framework Foundation"
            "-framework Metal"
            "-framework MetalKit"
        )
    endif()

    # Vulkan
    if(_GGML_VULKAN_ENABLED)
        target_link_libraries(3rdparty_ggml INTERFACE
            ${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-vulkan${_GGML_LIB_SUFFIX}
            Vulkan::Vulkan
        )
    endif()

    # OpenCL
    if(_GGML_OPENCL_ENABLED)
        if(NOT TARGET OpenCL::OpenCL)
            if(OpenCL_LIBRARIES AND OpenCL_INCLUDE_DIRS)
                add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
                set_target_properties(OpenCL::OpenCL PROPERTIES
                    IMPORTED_LOCATION "${OpenCL_LIBRARIES}"
                    INTERFACE_INCLUDE_DIRECTORIES "${OpenCL_INCLUDE_DIRS}")
            else()
                find_library(_GGML_OPENCL_LIB OpenCL)
                if(_GGML_OPENCL_LIB)
                    add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
                    set_target_properties(OpenCL::OpenCL PROPERTIES
                        IMPORTED_LOCATION "${_GGML_OPENCL_LIB}")
                endif()
            endif()
        endif()
        target_link_libraries(3rdparty_ggml INTERFACE
            ${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-opencl${_GGML_LIB_SUFFIX}
        )
        if(TARGET OpenCL::OpenCL)
            target_link_libraries(3rdparty_ggml INTERFACE OpenCL::OpenCL)
        elseif(OpenCL_LIBRARIES)
            target_link_libraries(3rdparty_ggml INTERFACE ${OpenCL_LIBRARIES})
        endif()
    endif()

else()
    # SHARED + GGML_BACKEND_DL mode: all configured backends (Metal, CUDA, OpenCL,
    # Vulkan) are standalone dynamic modules loaded at runtime via
    # ggml_backend_load_all_from_path(). They are NOT linked into
    # 3rdparty_ggml. Each backend module links its own platform dependencies.
    # CopyGgmlBackends.cmake copies them alongside libAICore, and
    # load_backends_once() resolves the search directory at runtime.
endif()

# OpenMP (ggml defaults GGML_OPENMP=ON; link the runtime when available)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(3rdparty_ggml INTERFACE OpenMP::OpenMP_CXX)
endif()

# --- Platform dependencies ---
find_package(Threads REQUIRED)
target_link_libraries(3rdparty_ggml INTERFACE Threads::Threads)

if(UNIX AND NOT APPLE)
    target_link_libraries(3rdparty_ggml INTERFACE dl m)
endif()

# --- Export variables for downstream consumers ---
set(GGML_FOUND TRUE CACHE BOOL "ggml found" FORCE)
set(GGML_INCLUDE_DIRS ${GGML_INCLUDE_DIRS} CACHE PATH "ggml include dirs" FORCE)
set(GGML_LIBRARIES 3rdparty_ggml CACHE STRING "ggml libraries" FORCE)
set(GGML_CUDA_ENABLED ${_GGML_CUDA_ENABLED} CACHE BOOL "ggml CUDA backend built" FORCE)
set(GGML_BUNDLE_CUDA_RUNTIME ${GGML_BUNDLE_CUDA_RUNTIME} CACHE BOOL
    "Bundle CUDA runtime into installer for driver-only deployment" FORCE)
set(GGML_VULKAN_ENABLED ${_GGML_VULKAN_ENABLED} CACHE BOOL
    "Read-only result: ON only when GGML_USE_VULKAN=ON and glslc/Vulkan/SPIR-V deps were found" FORCE)
mark_as_advanced(GGML_VULKAN_ENABLED)
set(GGML_OPENCL_ENABLED ${_GGML_OPENCL_ENABLED} CACHE BOOL "ggml OpenCL backend built" FORCE)
set(GGML_SYCL_ENABLED ${_GGML_SYCL_ENABLED} CACHE BOOL "ggml SYCL backend built" FORCE)
set(GGML_METAL_ENABLED ${_GGML_METAL_ENABLED} CACHE BOOL "ggml Metal backend built" FORCE)
set(GGML_DYNAMIC_BACKENDS ${GGML_BUILD_SHARED} CACHE BOOL "ggml backends are dynamic modules" FORCE)
set(GGML_RUNTIME_LIB_DIR "${GGML_RUNTIME_LIB_DIR}" CACHE PATH
    "ggml core runtime library directory" FORCE)
set(GGML_BACKEND_DIR "${GGML_MODULE_DIR}" CACHE PATH
    "ggml dynamic backend module directory" FORCE)
set(GGML_MODULE_SUFFIX "${_GGML_MODULE_SUFFIX}" CACHE STRING
    "ggml dynamic backend module suffix" FORCE)

set(GGML_CPU_ALL_VARIANTS ${GGML_CPU_ALL_VARIANTS} CACHE BOOL
    "ggml CPU micro-architecture variants built" FORCE)
set(_ggml_gpu_backend_module_files)
foreach(_backend IN ITEMS cuda metal vulkan opencl sycl)
    string(TOUPPER "${_backend}" _backend_upper)
    if(_GGML_${_backend_upper}_ENABLED)
        list(APPEND _ggml_gpu_backend_module_files
            "${CMAKE_SHARED_LIBRARY_PREFIX}ggml-${_backend}${_GGML_MODULE_SUFFIX}")
    endif()
endforeach()
list(REMOVE_DUPLICATES _ggml_gpu_backend_module_files)
set(GGML_GPU_BACKEND_MODULE_FILES "${_ggml_gpu_backend_module_files}" CACHE STRING
    "Configured ggml GPU backend module files" FORCE)
set(_ggml_backend_module_files)
if(GGML_CPU_ALL_VARIANTS)
    set(_ggml_backend_module_files ${_ggml_gpu_backend_module_files})
else()
    list(APPEND _ggml_backend_module_files
        "${CMAKE_SHARED_LIBRARY_PREFIX}ggml-cpu${_GGML_MODULE_SUFFIX}")
    list(APPEND _ggml_backend_module_files ${_ggml_gpu_backend_module_files})
endif()
list(REMOVE_DUPLICATES _ggml_backend_module_files)
set(GGML_BACKEND_MODULE_FILES "${_ggml_backend_module_files}" CACHE STRING
    "Configured ggml runtime backend module files (CPU via glob when ALL_VARIANTS)" FORCE)

set(_GGML_BACKEND_LIST "CPU")
if(_GGML_CUDA_ENABLED)
    list(APPEND _GGML_BACKEND_LIST "CUDA")
endif()
if(_GGML_OPENCL_ENABLED)
    list(APPEND _GGML_BACKEND_LIST "OpenCL")
endif()
if(_GGML_SYCL_ENABLED)
    list(APPEND _GGML_BACKEND_LIST "SYCL")
endif()
if(_GGML_VULKAN_ENABLED)
    list(APPEND _GGML_BACKEND_LIST "Vulkan")
endif()
if(_GGML_METAL_ENABLED)
    list(APPEND _GGML_BACKEND_LIST "Metal")
endif()
set(_GGML_AUTO_ORDER_LIST)
if(_GGML_CUDA_ENABLED)
    list(APPEND _GGML_AUTO_ORDER_LIST "CUDA")
endif()
if(APPLE)
    if(_GGML_METAL_ENABLED)
        list(APPEND _GGML_AUTO_ORDER_LIST "Metal")
    endif()
else()
    if(_GGML_VULKAN_ENABLED)
        list(APPEND _GGML_AUTO_ORDER_LIST "Vulkan")
    endif()
endif()
list(APPEND _GGML_AUTO_ORDER_LIST "CPU")
string(JOIN " -> " _GGML_AUTO_ORDER ${_GGML_AUTO_ORDER_LIST})
string(JOIN ", " _GGML_BACKEND_LIST_TEXT ${_GGML_BACKEND_LIST})
set(GGML_BACKEND_LIST "${_GGML_BACKEND_LIST_TEXT}" CACHE STRING
    "ggml backends built at configure time" FORCE)
set(GGML_AUTO_BACKEND_ORDER "${_GGML_AUTO_ORDER}" CACHE STRING
    "ggml Auto device selection order at runtime" FORCE)
if(GGML_BUILD_SHARED)
    set(_GGML_LINK_MODE "shared (dynamic backend loading)")
else()
    set(_GGML_LINK_MODE "static")
endif()
set(_GGML_STATUS_SUFFIX "")
if(GGML_CPU_ALL_VARIANTS)
    set(_GGML_STATUS_SUFFIX ", CPU all variants ON")
endif()
message(STATUS "ggml ${GGML_VERSION}: backends = ${_GGML_BACKEND_LIST} "
                "(${_GGML_LINK_MODE}, auto runtime order: ${_GGML_AUTO_ORDER}${_GGML_STATUS_SUFFIX})")
message(STATUS "ggml ${GGML_VERSION}: install=${GGML_INSTALL_DIR}, download_dir=${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/ggml")
