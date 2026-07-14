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

# CUDA backend
option(GGML_USE_CUDA "Enable ggml CUDA backend" OFF)
if(BUILD_CUDA_MODULE OR GGML_USE_CUDA)
    list(APPEND GGML_CMAKE_ARGS -DGGML_CUDA=ON)
    if(DEFINED CUDAToolkit_ROOT)
        list(APPEND GGML_CMAKE_ARGS -DCUDAToolkit_ROOT=${CUDAToolkit_ROOT})
    endif()

    # Reuse project-wide CUDA architecture selection so that ggml compiles
    # the same set of GPU kernels as the rest of ACloudViewer.
    include(${CMAKE_SOURCE_DIR}/cmake/CloudViewerMakeCudaArchitectures.cmake)
    cloudViewer_make_cuda_architectures(_ggml_cuda_archs)
    if(_ggml_cuda_archs)
        # ExternalProject needs semicolons escaped in list arguments
        string(REPLACE ";" "$<SEMICOLON>" _ggml_cuda_archs_escaped "${_ggml_cuda_archs}")
        list(APPEND GGML_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=${_ggml_cuda_archs_escaped})
        message(STATUS "ggml: CUDA architectures = ${_ggml_cuda_archs}")
    else()
        list(APPEND GGML_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=native)
        message(STATUS "ggml: CUDA architectures = native (fallback)")
    endif()
    list(APPEND GGML_CMAKE_ARGS -DGGML_CUDA_F16=ON)
    message(STATUS "ggml: CUDA backend enabled")
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_CUDA=OFF)
endif()

# Metal backend (Apple only)
option(GGML_USE_METAL "Enable ggml Metal backend" OFF)
if(APPLE OR GGML_USE_METAL)
    list(APPEND GGML_CMAKE_ARGS -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON)
    message(STATUS "ggml: Metal backend enabled")
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_METAL=OFF)
endif()

# Vulkan backend
option(GGML_USE_VULKAN "Enable ggml Vulkan backend" OFF)
if(GGML_USE_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        list(APPEND GGML_CMAKE_ARGS -DGGML_VULKAN=ON)
        message(STATUS "ggml: Vulkan backend enabled")
    else()
        message(WARNING "ggml: Vulkan requested but not found, disabling")
        list(APPEND GGML_CMAKE_ARGS -DGGML_VULKAN=OFF)
    endif()
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_VULKAN=OFF)
endif()

# OpenCL backend
option(GGML_USE_OPENCL "Enable ggml OpenCL backend" OFF)
if(GGML_USE_OPENCL)
    list(APPEND GGML_CMAKE_ARGS -DGGML_OPENCL=ON)
    message(STATUS "ggml: OpenCL backend enabled")
else()
    list(APPEND GGML_CMAKE_ARGS -DGGML_OPENCL=OFF)
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
if(BUILD_CUDA_MODULE OR GGML_USE_CUDA)
    list(APPEND GGML_BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-cuda${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
endif()
if(APPLE OR GGML_USE_METAL)
    list(APPEND GGML_BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-metal${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
endif()
if(GGML_USE_VULKAN AND Vulkan_FOUND)
    list(APPEND GGML_BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-vulkan${CMAKE_STATIC_LIBRARY_SUFFIX}
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
if(BUILD_CUDA_MODULE OR GGML_USE_CUDA)
    target_link_libraries(3rdparty_ggml INTERFACE
        ${GGML_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-cuda${CMAKE_STATIC_LIBRARY_SUFFIX}
        CUDA::cuda_driver
        CUDA::cudart
    )
endif()

# Metal (macOS/iOS)
if(APPLE OR GGML_USE_METAL)
    target_link_libraries(3rdparty_ggml INTERFACE
        ${GGML_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-metal${CMAKE_STATIC_LIBRARY_SUFFIX}
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalKit"
    )
endif()

# Vulkan
if(GGML_USE_VULKAN AND Vulkan_FOUND)
    target_link_libraries(3rdparty_ggml INTERFACE
        ${GGML_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-vulkan${CMAKE_STATIC_LIBRARY_SUFFIX}
        Vulkan::Vulkan
    )
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

message(STATUS "ggml ${GGML_VERSION}: install=${GGML_INSTALL_DIR}, download_dir=${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/ggml")
