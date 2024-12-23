# Exports: ${STDGPU_INCLUDE_DIRS}
# Exports: ${STDGPU_LIB_DIR}
# Exports: ${STDGPU_LIBRARIES}

include(ExternalProject)

ExternalProject_Add(
    ext_stdgpu
    PREFIX stdgpu
    # URL https://github.com/stotko/stdgpu/archive/e10f6f3ccc9902d693af4380c3bcd188ec34a2e6.tar.gz
    # URL_HASH SHA256=7bb2733b099f7cedc86d2aee7830d128ac1222cfafa34cbaa4e818483c0a93f6
    # Jun 20 2024. Later versions need CUDA 11.5 and an API update (stdgpu::pair)
    URL https://github.com/stotko/stdgpu/archive/1b6a3319f1fbf180166e1bbc1d75f69ab622a0a0.tar.gz
    URL_HASH SHA256=faa3bf9cbe49ef9cc09e2e07e60d10bbf3b896edb6089c920bebe0f850fd95e4
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/stdgpu"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCUDAToolkit_ROOT=${CUDAToolkit_LIBRARY_ROOT}
        -DSTDGPU_BUILD_SHARED_LIBS=OFF
        -DSTDGPU_BUILD_EXAMPLES=OFF
        -DSTDGPU_BUILD_TESTS=OFF
        -DSTDGPU_BUILD_BENCHMARKS=OFF
        -DSTDGPU_ENABLE_CONTRACT_CHECKS=OFF
        -DTHRUST_INCLUDE_DIR=${CUDAToolkit_INCLUDE_DIRS}
        ${ExternalProject_CMAKE_ARGS_hidden}
    CMAKE_CACHE_ARGS    # Lists must be passed via CMAKE_CACHE_ARGS
        -DCMAKE_CUDA_ARCHITECTURES:STRING=${CMAKE_CUDA_ARCHITECTURES}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}stdgpu${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_stdgpu INSTALL_DIR)
set(STDGPU_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(STDGPU_LIB_DIR ${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR})
set(STDGPU_LIBRARIES stdgpu)
