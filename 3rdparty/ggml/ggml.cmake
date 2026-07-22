# Exports: ${GGML_INCLUDE_DIRS}
# Exports: ${GGML_LIB_DIR}
# Exports: ${GGML_LIBRARIES}

include(ExternalProject)

option(GGML_ENABLED "Enable ggml inference library for AI model support" ON)

if(NOT GGML_ENABLED)
    message(STATUS "ggml is disabled")
    return()
endif()

set(GGML_VERSION "0.15.1")
set(GGML_URL "https://github.com/ggml-org/ggml/archive/refs/tags/v${GGML_VERSION}.tar.gz")
set(GGML_SHA256 "b2fd615a552c0aeba35be361fd7e59c55623c94bffe5ca1acc5162e5d98e15ec")

# Reuse project-wide ExternalProject_CMAKE_ARGS_hidden (includes compiler,
# build type, MSVC runtime, PIC, visibility, CUDA compiler, etc.)
set(GGML_CMAKE_ARGS
    ${ExternalProject_CMAKE_ARGS_hidden}
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_BUILD_TESTS=OFF
    -DGGML_BUILD_EXAMPLES=OFF
    -DGGML_STATIC=ON
)

# --- Backend selection ---
# ggml can compile multiple backends into one build (CUDA + OpenCL + Vulkan + CPU).
# CPU is always built. GPU backends below are enabled when ON and dependencies are found.
#
# Platform defaults (override with -DGGML_USE_*=ON/OFF):
#   Apple:     Metal ON, Vulkan OFF, OpenCL OFF
#   Linux/Win: Metal OFF, Vulkan OFF, OpenCL ON  (auto-detect when dependencies exist)
if(APPLE)
    set(_GGML_METAL_DEFAULT ON)
    set(_GGML_OPENCL_DEFAULT OFF)
elseif(WIN32 OR UNIX)
    set(_GGML_METAL_DEFAULT OFF)
    set(_GGML_OPENCL_DEFAULT ON)
else()
    set(_GGML_METAL_DEFAULT OFF)
    set(_GGML_OPENCL_DEFAULT OFF)
endif()

# CUDA backend (opt-in: NVIDIA toolchain required)
option(GGML_USE_CUDA "Enable ggml CUDA backend" OFF)
set(_GGML_CUDA_ENABLED OFF)
if(BUILD_CUDA_MODULE OR GGML_USE_CUDA)
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

# Metal backend (Apple; default ON on macOS/iOS)
option(GGML_USE_METAL "Enable ggml Metal backend (default ON on Apple)" ${_GGML_METAL_DEFAULT})
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

# Vulkan backend (opt-in: requires Vulkan SDK + glslc + SPIRV-Headers)
option(GGML_USE_VULKAN "Enable ggml Vulkan backend (opt-in; requires Vulkan SDK)" OFF)
set(_GGML_VULKAN_ENABLED OFF)
if(GGML_USE_VULKAN)
    if(DEFINED ENV{VULKAN_SDK})
        list(APPEND CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK}")
    endif()
    find_package(Vulkan COMPONENTS glslc QUIET)
    find_package(SPIRV-Headers CONFIG QUIET)
    if(Vulkan_FOUND AND Vulkan_GLSLC_EXECUTABLE AND SPIRV-Headers_FOUND)
        list(APPEND GGML_CMAKE_ARGS -DGGML_VULKAN=ON)
        if(SPIRV-Headers_DIR)
            list(APPEND GGML_CMAKE_ARGS -DSPIRV-Headers_DIR=${SPIRV-Headers_DIR})
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
        message(STATUS "ggml: Vulkan backend enabled (glslc=${Vulkan_GLSLC_EXECUTABLE})")
    else()
        list(APPEND GGML_CMAKE_ARGS -DGGML_VULKAN=OFF)
        if(NOT Vulkan_FOUND)
            message(STATUS "ggml: Vulkan loader not found — skipping Vulkan backend")
        elseif(NOT Vulkan_GLSLC_EXECUTABLE)
            message(STATUS "ggml: glslc not found — skipping Vulkan backend "
                            "(install Vulkan SDK or vulkan-tools)")
        elseif(NOT SPIRV-Headers_FOUND)
            message(STATUS "ggml: SPIRV-Headers not found — skipping Vulkan backend "
                            "(install spirv-headers / Vulkan SDK, or set VULKAN_SDK)")
        endif()
    endif()
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_VULKAN=OFF)
endif()

# OpenCL backend — default ON on Linux/Windows, OFF on macOS; auto-detect OpenCL 3.0 + Python3
option(GGML_USE_OPENCL "Enable ggml OpenCL backend (default ON on Linux/Windows; auto-detect when ON)" ${_GGML_OPENCL_DEFAULT})
set(_GGML_OPENCL_ENABLED OFF)
if(GGML_USE_OPENCL)
    if(APPLE)
        list(APPEND GGML_CMAKE_ARGS -DGGML_OPENCL=OFF)
        message(STATUS "ggml: OpenCL disabled on macOS (system OpenCL is deprecated and unusable for ggml; use Metal)")
    else()
        find_package(OpenCL QUIET)
        find_package(Python3 QUIET COMPONENTS Interpreter)
        set(_GGML_OPENCL_HEADERS_OK OFF)
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
                if(_ggml_opencl_cl_h MATCHES "#define CL_VERSION_3_0"
                   OR _ggml_opencl_ver_h MATCHES "#define CL_VERSION_3_0")
                    set(_GGML_OPENCL_HEADERS_OK ON)
                endif()
            endif()
        endif()
        if(OpenCL_FOUND AND Python3_FOUND AND _GGML_OPENCL_HEADERS_OK)
            list(APPEND GGML_CMAKE_ARGS -DGGML_OPENCL=ON)
            set(_GGML_OPENCL_ENABLED ON)
            message(STATUS "ggml: OpenCL backend enabled (OpenCL 3.0 headers)")
        else()
            list(APPEND GGML_CMAKE_ARGS -DGGML_OPENCL=OFF)
            if(NOT OpenCL_FOUND)
                message(STATUS "ggml: OpenCL not found — skipping OpenCL backend "
                                "(install ocl-icd-opencl-dev / OpenCL ICD)")
            elseif(NOT _GGML_OPENCL_HEADERS_OK)
                message(STATUS "ggml: OpenCL 3.0 headers not found — skipping OpenCL backend "
                                "(install Khronos OpenCL-Headers; Ubuntu 20.04 apt only ships 2.2)")
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

# BLAS / Accelerate backend
option(GGML_USE_BLAS "Enable ggml BLAS backend" OFF)
if(GGML_USE_BLAS OR APPLE)
    list(APPEND GGML_CMAKE_ARGS -DGGML_BLAS=ON)
    if(APPLE)
        list(APPEND GGML_CMAKE_ARGS -DGGML_ACCELERATE=ON)
    endif()
    message(STATUS "ggml: BLAS backend enabled")
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_BLAS=OFF)
endif()

# tinyBLAS (llamafile) for optimized CPU matmul
option(GGML_USE_LLAMAFILE "Enable ggml tinyBLAS (llamafile) matmul kernels" ON)
list(APPEND GGML_CMAKE_ARGS -DGGML_LLAMAFILE=${GGML_USE_LLAMAFILE})

# CPU optimizations
if(NOT CMAKE_CROSSCOMPILING)
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
set(GGML_BUILD_BYPRODUCTS
    <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml${CMAKE_STATIC_LIBRARY_SUFFIX}
    <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-base${CMAKE_STATIC_LIBRARY_SUFFIX}
    <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-cpu${CMAKE_STATIC_LIBRARY_SUFFIX}
)
if(_GGML_CUDA_ENABLED)
    list(APPEND GGML_BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-cuda${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
endif()
if(_GGML_METAL_ENABLED)
    list(APPEND GGML_BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-metal${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
endif()
if(GGML_USE_BLAS OR APPLE)
    list(APPEND GGML_BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-blas${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
endif()
if(_GGML_VULKAN_ENABLED)
    list(APPEND GGML_BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-vulkan${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
endif()
if(_GGML_OPENCL_ENABLED)
    list(APPEND GGML_BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-opencl${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
endif()

ExternalProject_Add(ext_ggml
    PREFIX ggml
    URL ${GGML_URL}
    URL_HASH SHA256=${GGML_SHA256}
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/ggml"
    UPDATE_COMMAND ""
    CMAKE_ARGS ${GGML_CMAKE_ARGS}
    BUILD_BYPRODUCTS ${GGML_BUILD_BYPRODUCTS}
)

ExternalProject_Get_Property(ext_ggml INSTALL_DIR)
set(GGML_INSTALL_DIR ${INSTALL_DIR})

set(GGML_INCLUDE_DIRS ${GGML_INSTALL_DIR}/include)
set(GGML_LIB_DIR ${GGML_INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR})

# --- Create imported interface target ---
add_library(3rdparty_ggml INTERFACE)
add_dependencies(3rdparty_ggml ext_ggml)

target_include_directories(3rdparty_ggml INTERFACE ${GGML_INCLUDE_DIRS})

# Link order matters: ggml -> ggml-cpu -> ggml-base
target_link_libraries(3rdparty_ggml INTERFACE
    ${GGML_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${GGML_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-cpu${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${GGML_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-base${CMAKE_STATIC_LIBRARY_SUFFIX}
)

# --- GPU backend linking ---

# CUDA: reuse CUDAToolkit already found by the parent project
if(_GGML_CUDA_ENABLED)
    target_link_libraries(3rdparty_ggml INTERFACE
        ${GGML_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-cuda${CMAKE_STATIC_LIBRARY_SUFFIX}
        CUDA::cuda_driver
        CUDA::cudart
    )
    # ggml-cuda references cublas; link shared libs after the static archive.
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
        ${GGML_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-metal${CMAKE_STATIC_LIBRARY_SUFFIX}
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalKit"
    )
endif()

# Vulkan
if(_GGML_VULKAN_ENABLED)
    target_link_libraries(3rdparty_ggml INTERFACE
        ${GGML_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-vulkan${CMAKE_STATIC_LIBRARY_SUFFIX}
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
        ${GGML_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-opencl${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
    if(TARGET OpenCL::OpenCL)
        target_link_libraries(3rdparty_ggml INTERFACE OpenCL::OpenCL)
    elseif(OpenCL_LIBRARIES)
        target_link_libraries(3rdparty_ggml INTERFACE ${OpenCL_LIBRARIES})
    endif()
endif()

# BLAS / Accelerate
if(GGML_USE_BLAS OR APPLE)
    target_link_libraries(3rdparty_ggml INTERFACE
        ${GGML_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-blas${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
    if(APPLE)
        target_link_libraries(3rdparty_ggml INTERFACE "-framework Accelerate")
    else()
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            target_link_libraries(3rdparty_ggml INTERFACE ${BLAS_LIBRARIES})
        endif()
    endif()
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
set(GGML_VULKAN_ENABLED ${_GGML_VULKAN_ENABLED} CACHE BOOL "ggml Vulkan backend built" FORCE)
set(GGML_OPENCL_ENABLED ${_GGML_OPENCL_ENABLED} CACHE BOOL "ggml OpenCL backend built" FORCE)
set(GGML_METAL_ENABLED ${_GGML_METAL_ENABLED} CACHE BOOL "ggml Metal backend built" FORCE)

set(_GGML_BACKEND_LIST "CPU")
if(_GGML_CUDA_ENABLED)
    list(APPEND _GGML_BACKEND_LIST "CUDA")
endif()
if(_GGML_OPENCL_ENABLED)
    list(APPEND _GGML_BACKEND_LIST "OpenCL")
endif()
if(_GGML_VULKAN_ENABLED)
    list(APPEND _GGML_BACKEND_LIST "Vulkan")
endif()
if(_GGML_METAL_ENABLED)
    list(APPEND _GGML_BACKEND_LIST "Metal")
endif()
if(APPLE)
    set(_GGML_AUTO_ORDER "Metal -> CUDA -> CPU")
else()
    set(_GGML_AUTO_ORDER "CUDA -> OpenCL -> CPU")
endif()
message(STATUS "ggml ${GGML_VERSION}: backends = ${_GGML_BACKEND_LIST} "
                "(auto runtime order: ${_GGML_AUTO_ORDER})")
message(STATUS "ggml ${GGML_VERSION}: install=${GGML_INSTALL_DIR}, download_dir=${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/ggml")
