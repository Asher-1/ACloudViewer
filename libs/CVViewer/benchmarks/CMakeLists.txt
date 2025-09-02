add_executable(benchmarks)

include_directories(${CloudViewer_SOURCE_DIR})

add_subdirectory(benchmark_utilities)
add_subdirectory(core)
add_subdirectory(geometry)
add_subdirectory(io)
add_subdirectory(pipelines)
add_subdirectory(t/geometry)
add_subdirectory(t/io)
add_subdirectory(t/pipelines)

target_link_libraries(benchmarks PRIVATE CloudViewer::CloudViewer)

if (BUILD_CUDA_MODULE)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(benchmarks PRIVATE CUDA::cudart)
endif()

cloudViewer_show_and_abort_on_warning(benchmarks)
cloudViewer_set_global_properties(benchmarks)
cloudViewer_link_3rdparty_libraries(benchmarks)

# fix that You must build your code with position independent
# code if Qt was built with -reduce-relocations
cloudViewer_set_targets_independent(benchmarks)

if (BUILD_CUDA_MODULE)
    target_include_directories(benchmarks SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if (WITH_IPP)
    target_compile_definitions(benchmarks PRIVATE IPP_CONDITIONAL_TEST_STR=) # Empty string (test not disabled)
else()
    target_compile_definitions(benchmarks PRIVATE IPP_CONDITIONAL_TEST_STR=DISABLED_)
endif()

target_sources(benchmarks PRIVATE
    Main.cpp
)
target_link_libraries(benchmarks PRIVATE benchmark::benchmark)
target_include_directories(benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})