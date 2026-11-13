cmake_minimum_required(VERSION 3.24)

project(AICoreCuMeshBuildSmoke LANGUAGES CXX CUDA)
enable_testing()

find_package(CUDAToolkit REQUIRED)

option(CUMESH_SMOKE_REQUIRE_ENABLED
    "Fail when CuMesh prerequisites are unavailable" ON)
option(CUMESH_SMOKE_GPU "Run CuMesh mesh operations on an available CUDA device" OFF)

set(AICore_USE_CUMESH ON)
set(AICore_CUDA_ENABLED ON)
set(CLOUDVIEWER_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/CloudViewer")
set(CloudViewer_INSTALL_LIB_DIR lib)

set(_repo_root "${CMAKE_CURRENT_LIST_DIR}/../../../..")
set(_trellis_src_dir "${_repo_root}/core/AICore/src/tasks/trellis")

add_executable(aicore_cumesh_smoke cumesh_link_probe.cpp)
set_target_properties(aicore_cumesh_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_include_directories(aicore_cumesh_smoke PRIVATE
    "${_trellis_src_dir}/third_party/cumesh_glue")

function(cloudViewer_install_targets target)
    install(TARGETS ${target} RUNTIME DESTINATION bin LIBRARY DESTINATION lib
            COMPONENT Runtime)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CLOUDVIEWER_OUTPUT_DIRECTORY}"
        COMMAND ${CMAKE_COMMAND} -E copy
            "$<TARGET_FILE:${target}>"
            "${CLOUDVIEWER_OUTPUT_DIRECTORY}/"
        VERBATIM)
endfunction()
function(InstallSharedLibrary)
endfunction()

include("${_trellis_src_dir}/cmake/cumesh.cmake")
set(_smoke_cuda_architectures "${CMAKE_CUDA_ARCHITECTURES}")
aicore_trellis_wire_cumesh(aicore_cumesh_smoke "${_trellis_src_dir}")

if(NOT TARGET aicore_cumesh)
    if(CUMESH_SMOKE_REQUIRE_ENABLED)
        message(FATAL_ERROR "CuMesh was not enabled by the build smoke configuration")
    endif()
    message(STATUS "CuMesh correctly disabled by an unavailable prerequisite")
else()
    get_target_property(_actual_cuda_architectures aicore_cumesh CUDA_ARCHITECTURES)
    if(NOT "${_actual_cuda_architectures}" STREQUAL "${_smoke_cuda_architectures}")
        message(FATAL_ERROR "CuMesh did not preserve the configured CUDA architectures")
    endif()
    add_test(NAME cumesh_link_contract COMMAND aicore_cumesh_smoke)
    if(CUMESH_SMOKE_GPU)
        add_test(NAME cumesh_gpu_mesh COMMAND aicore_cumesh_smoke --gpu)
    endif()

    # Bit-level parity of remove_duplicate_faces against a CPU reference that
    # reproduces the three-pass radix order and the rewritten in-place scan.
    # The CuMesh sources are compiled into the test rather than linked from
    # aicore_cumesh: that library exports only the C glue, so the C++ methods
    # are not available to a consumer on platforms with hidden visibility.
    add_executable(aicore_cumesh_dedup_parity cumesh_dedup_parity.cu
        "${_trellis_src_dir}/third_party/CuMesh/src/atlas.cu"
        "${_trellis_src_dir}/third_party/CuMesh/src/clean_up.cu"
        "${_trellis_src_dir}/third_party/CuMesh/src/connectivity.cu"
        "${_trellis_src_dir}/third_party/CuMesh/src/cumesh.cu"
        "${_trellis_src_dir}/third_party/CuMesh/src/geometry.cu"
        "${_trellis_src_dir}/third_party/CuMesh/src/io.cu"
        "${_trellis_src_dir}/third_party/CuMesh/src/shared.cu")
    set_target_properties(aicore_cumesh_dedup_parity PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
        CUDA_RUNTIME_LIBRARY Static
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    target_include_directories(aicore_cumesh_dedup_parity PRIVATE
        "${_trellis_src_dir}/../.."
        "${_trellis_src_dir}/third_party/CuMesh/src"
        "${_trellis_src_dir}")
    target_compile_options(aicore_cumesh_dedup_parity PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
    target_link_libraries(aicore_cumesh_dedup_parity PRIVATE
                          CUDA::cudart_static)
    if(NOT "${_smoke_cuda_architectures}" STREQUAL "")
        set_target_properties(aicore_cumesh_dedup_parity PROPERTIES
            CUDA_ARCHITECTURES "${_smoke_cuda_architectures}")
    endif()
    if(CUMESH_SMOKE_GPU)
        add_test(NAME cumesh_dedup_parity COMMAND aicore_cumesh_dedup_parity)
    endif()
endif()

if(WIN32 AND TARGET aicore_cumesh)
    add_custom_command(TARGET aicore_cumesh_smoke POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            "$<TARGET_FILE:aicore_cumesh_smoke>"
            "${CLOUDVIEWER_OUTPUT_DIRECTORY}/"
        VERBATIM)
endif()
