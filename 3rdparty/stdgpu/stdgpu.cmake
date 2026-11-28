# Exports: ${STDGPU_INCLUDE_DIRS}
# Exports: ${STDGPU_LIB_DIR}
# Exports: ${STDGPU_LIBRARIES}

include(ExternalProject)

find_package(Git QUIET REQUIRED)

# CUDAToolkit_INCLUDE_DIRS is a list. Since CUDA 13.0 the CCCL headers
# (thrust/cub/libcudacxx) moved under <toolkit>/include/cccl, so nvcc reports
# both "include" and "include/cccl". Passing the whole list as a single
# -DTHRUST_INCLUDE_DIR splits it on the semicolon: CMake keeps only the first
# entry ("include", which no longer holds thrust/version.h) and drops the cccl
# path as an "extra path", so stdgpu's Findthrust fails. Select the one
# directory that actually contains thrust/version.h ("include/cccl" on CUDA 13+,
# "include" on older toolkits).
set(STDGPU_THRUST_INCLUDE_DIR "")
foreach(_stdgpu_inc IN LISTS CUDAToolkit_INCLUDE_DIRS)
    if(EXISTS "${_stdgpu_inc}/thrust/version.h")
        set(STDGPU_THRUST_INCLUDE_DIR "${_stdgpu_inc}")
        break()
    endif()
endforeach()

# CCCL 3.x (CUDA 13.0+) removed thrust::detail::is_proxy_reference, which the
# pinned stdgpu specializes for its insert-iterator proxies; the trait is now
# is_wrapped_reference. Rename the three specializations so stdgpu compiles
# against CUDA 13. CUDA 11.8/12.x still provide is_proxy_reference, so the
# patch is applied only for CUDA 13+ (same approach as Open3D's stdgpu recipe).
# CUDA 13 also removed cudaDeviceProp::clockRate, which device.cpp prints;
# fix-cuda13-clock-rate.patch queries cudaDeviceGetAttribute instead (the
# attribute API exists on every supported toolkit, verified against 12.6).
# CCCL 3.x eagerly instantiates std::iterator_traits<thrust::pointer<void>>
# through its type traits; the pinned stdgpu's std::iterator_traits
# specializations then form std::iterator_traits<void*>::reference (void&),
# a hard error on GCC 15. fix-cuda13-iterator-traits.patch removes those
# specializations (upstream stdgpu 025465a77b "Add CUDA 13 support").
# CUDA 11.8/12.x thrust has no own std traits for thrust::pointer, so the
# removal stays gated to CUDA 13+ like the other two patches.
set(STDGPU_PATCH_COMMAND "")
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    set(STDGPU_PATCH_COMMAND
        ${CMAKE_COMMAND} -E chdir <SOURCE_DIR> ${GIT_EXECUTABLE} init
        COMMAND ${CMAKE_COMMAND} -E chdir <SOURCE_DIR> ${GIT_EXECUTABLE} add -A
        COMMAND ${CMAKE_COMMAND} -E chdir <SOURCE_DIR> ${GIT_EXECUTABLE} apply
            --ignore-space-change --ignore-whitespace
            ${CMAKE_CURRENT_LIST_DIR}/fix-thrust-is-proxy-reference.patch
            ${CMAKE_CURRENT_LIST_DIR}/fix-cuda13-clock-rate.patch
            ${CMAKE_CURRENT_LIST_DIR}/fix-cuda13-iterator-traits.patch
    )
endif()

ExternalProject_Add(
    ext_stdgpu
    PREFIX stdgpu
    URL https://github.com/stotko/stdgpu/archive/2588168d226bd17229dbf58d821549580791089d.tar.gz
    URL_HASH SHA256=86e50789bbe21c57f64358c6acbd4481d56c1e45ce9ba1fb5c5c8482c3973215
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/stdgpu"
    UPDATE_COMMAND ""
    PATCH_COMMAND ${STDGPU_PATCH_COMMAND}
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCUDAToolkit_ROOT=${CUDAToolkit_LIBRARY_ROOT}
        -DSTDGPU_BUILD_SHARED_LIBS=OFF
        -DSTDGPU_BUILD_EXAMPLES=OFF
        -DSTDGPU_BUILD_TESTS=OFF
        -DSTDGPU_BUILD_BENCHMARKS=OFF
        -DSTDGPU_ENABLE_CONTRACT_CHECKS=OFF
        -DTHRUST_INCLUDE_DIR=${STDGPU_THRUST_INCLUDE_DIR}
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
