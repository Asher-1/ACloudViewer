# Trellis CuMesh GPU chart clustering — task-private build wiring.
#
# Upstream trellis2cpp third_party/cumesh_glue integration (JeffreyXiang/
# CuMesh, MIT, vendored under src/tasks/trellis/third_party/CuMesh): GPU
# normal-cone chart clustering feeds xatlas' faceMaterialData as hard chart
# boundaries for the UV-atlas GLB bake (mesh_export.cpp's UnwrapMode::Auto).
#
# The vendored CuMesh core is plain CUDA/CUB: device memory is managed by
# cumesh::Buffer (cudaMalloc/cudaFree), every kernel and copy is ordered on the
# legacy default stream, and the glue exchanges host arrays only. There is no
# libtorch, Python, or pybind dependency. The target links the static CUDA
# runtime, so the installed library needs no CUDA redistributables.
#
# The user-facing switch is declared in cmake/AICoreOptions.cmake; everything
# below is task-owned. A missing prerequisite demotes the feature with a loud
# warning instead of failing configure:
#   CUDA off | CUDA toolkit < 11.4 | vendored CuMesh sources missing
#     → GPU clustering off; mesh_export tries CPU cone clustering, then
#       simple_unwrap if CPU clustering fails.

include_guard(GLOBAL)

function(aicore_trellis_wire_cumesh target trellis_src_dir)
    set(AICORE_CUMESH_ENABLED ${AICore_USE_CUMESH})
    # Preserve the consumer's native-CUDA architecture contract.
    get_target_property(_cumesh_cuda_architectures ${target} CUDA_ARCHITECTURES)
    if(NOT _cumesh_cuda_architectures
       OR _cumesh_cuda_architectures MATCHES "-NOTFOUND$")
        set(_cumesh_cuda_architectures "${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    # Guard ladder: each step may demote AICORE_CUMESH_ENABLED with a message.
    if(AICORE_CUMESH_ENABLED AND NOT AICore_CUDA_ENABLED)
        message(
            STATUS
                "AICore: CuMesh GPU chart clustering disabled (AICore_USE_CUDA=OFF)"
        )
        set(AICORE_CUMESH_ENABLED OFF)
    endif()
    if(AICORE_CUMESH_ENABLED)
        find_package(CUDAToolkit QUIET)
        if(NOT CUDAToolkit_FOUND)
            message(
                WARNING
                    "AICore: CuMesh needs the CUDA Toolkit — building without GPU chart clustering"
            )
            set(AICORE_CUMESH_ENABLED OFF)
        elseif(CUDAToolkit_VERSION VERSION_LESS 11.4)
            message(
                WARNING
                    "AICore: CuMesh needs CUDA Toolkit 11.4 or newer (found ${CUDAToolkit_VERSION}; the vendored kernels use the Ampere cp.async path and CUB device APIs) — building without GPU chart clustering"
            )
            set(AICORE_CUMESH_ENABLED OFF)
        endif()
    endif()
    if(AICORE_CUMESH_ENABLED
       AND NOT EXISTS "${trellis_src_dir}/third_party/CuMesh/src/cumesh.h")
        message(
            WARNING
                "AICore: missing vendored CuMesh at ${trellis_src_dir}/third_party/CuMesh — building without GPU chart clustering"
        )
        set(AICORE_CUMESH_ENABLED OFF)
    endif()
    if(NOT AICORE_CUMESH_ENABLED)
        return()
    endif()

    add_library(
        aicore_cumesh SHARED
        ${trellis_src_dir}/third_party/CuMesh/src/atlas.cu
        ${trellis_src_dir}/third_party/CuMesh/src/clean_up.cu
        ${trellis_src_dir}/third_party/CuMesh/src/connectivity.cu
        ${trellis_src_dir}/third_party/CuMesh/src/cumesh.cu
        ${trellis_src_dir}/third_party/CuMesh/src/geometry.cu
        ${trellis_src_dir}/third_party/CuMesh/src/io.cu
        ${trellis_src_dir}/third_party/CuMesh/src/shared.cu
        ${trellis_src_dir}/third_party/cumesh_glue/cumesh_glue.cu)
    target_compile_definitions(aicore_cumesh PRIVATE CUMESH_GLUE_BUILD=1)
    target_include_directories(
        aicore_cumesh
        PUBLIC "$<BUILD_INTERFACE:${trellis_src_dir}/../../../include>"
        PRIVATE ${trellis_src_dir}
                ${trellis_src_dir}/../..
                ${trellis_src_dir}/third_party/CuMesh/src
                ${trellis_src_dir}/third_party/cumesh_glue)
    set_target_properties(
        aicore_cumesh PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                                 CUDA_RUNTIME_LIBRARY Static
                                 POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(aicore_cumesh PRIVATE CUDA::cudart_static)
    if(NOT "${_cumesh_cuda_architectures}" STREQUAL "")
        set_target_properties(aicore_cumesh PROPERTIES
                              CUDA_ARCHITECTURES "${_cumesh_cuda_architectures}")
    endif()
    # The vendored kernels use constexpr helpers and extended lambdas.
    target_compile_options(
        aicore_cumesh PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

    get_target_property(_cumesh_runtime_dir ${target} RUNTIME_OUTPUT_DIRECTORY)
    get_target_property(_cumesh_library_dir ${target} LIBRARY_OUTPUT_DIRECTORY)
    if(NOT _cumesh_runtime_dir OR _cumesh_runtime_dir MATCHES "-NOTFOUND$")
        set(_cumesh_runtime_dir "${CMAKE_BINARY_DIR}/bin")
    endif()
    if(NOT _cumesh_library_dir OR _cumesh_library_dir MATCHES "-NOTFOUND$")
        set(_cumesh_library_dir "${CMAKE_BINARY_DIR}/bin")
    endif()
    set_target_properties(
        aicore_cumesh PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${_cumesh_runtime_dir}"
        LIBRARY_OUTPUT_DIRECTORY "${_cumesh_library_dir}")
    if(UNIX AND NOT APPLE)
        # The installed bridge sits beside libAICore.
        set_target_properties(aicore_cumesh PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()

    target_compile_definitions(${target} PRIVATE TRELLIS2_HAVE_CUMESH=1)
    target_include_directories(
        ${target} PRIVATE ${trellis_src_dir}/third_party/cumesh_glue)
    target_link_libraries(${target} PRIVATE aicore_cumesh)
    if(NOT APPLE)
        cloudViewer_install_targets(aicore_cumesh)
        InstallSharedLibrary(TARGET aicore_cumesh)
    endif()
    message(STATUS "AICore: CuMesh GPU chart clustering enabled")
endfunction()
