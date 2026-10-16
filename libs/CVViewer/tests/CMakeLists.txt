
# Overide policy for hidden symbol visibility
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

add_executable(tests)

add_subdirectory(camera)
add_subdirectory(core)
add_subdirectory(data)
add_subdirectory(geometry)
add_subdirectory(io)
add_subdirectory(ml)
add_subdirectory(t/geometry)
add_subdirectory(t/io)
add_subdirectory(t/pipelines)
add_subdirectory(test_utility)
add_subdirectory(utility)

target_sources(tests PRIVATE
    Main.cpp
    Tests.cpp
)

target_include_directories(tests PRIVATE ".")

# Monolithic test runner (500MB+ linked against the full library): keep it out
# of bin/ alongside the other grouped test folders (colmap_test/, aicore_tests/,
# plugin_test/).
# The per-config properties are load-bearing on MSVC:
# CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING (default ON) makes
# cloudviewer_apply_dev_runtime_bin_layout() set per-config variables
# (CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=bin/<Config>), and a per-config
# target property initialized from those variables OVERRIDES the generic
# RUNTIME_OUTPUT_DIRECTORY set here. The per-config values carry a $<CONFIG>
# generator expression because a per-config property is used verbatim — no
# configuration subdirectory is appended (measured: with a plain value the
# MSVC link emitted bin\cloudviewer_test\tests.exe while windows.yml locates
# the runner at bin\cloudviewer_test\<Config>\tests.exe, run 38852073162).
# Single-config Unix builds never see these properties, so the generic value
# keeps them at bin/cloudviewer_test/tests.
set_target_properties(tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/cloudviewer_test"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/cloudviewer_test/$<CONFIG>"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/cloudviewer_test/$<CONFIG>"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin/cloudviewer_test/$<CONFIG>")

# Fix Qt -fPIC requirement
cloudViewer_set_targets_independent(tests)

# If gpu not available, add "DISABLED_" to the gpu test names
if(BUILD_CUDA_MODULE)
    target_compile_definitions(tests PRIVATE GPU_CONDITIONAL_TEST_STR=) # Empty string
else()
    target_compile_definitions(tests PRIVATE GPU_CONDITIONAL_TEST_STR=DISABLED_)
endif()

if (WITH_IPP)
    target_compile_definitions(tests PRIVATE IPP_CONDITIONAL_TEST_STR=) # Empty string (test not disabled)
else()
    target_compile_definitions(tests PRIVATE IPP_CONDITIONAL_TEST_STR=DISABLED_)
endif()

target_link_libraries(tests PRIVATE
    CloudViewer::CloudViewer
    3rdparty_jsoncpp
    3rdparty_googletest
    3rdparty_threads
    3rdparty_vtk
)

if (TARGET 3rdparty_openmp)
    target_link_libraries(tests PRIVATE
        3rdparty_openmp
    )
endif()

cloudViewer_show_and_abort_on_warning(tests)
cloudViewer_set_global_properties(tests)

# On Windows, running tests from the build folder needs tbb.dll to be in the same folder.
if (WIN32)
  add_custom_command(
      TARGET tests
      POST_BUILD
      # $<TARGET_FILE_DIR:tests> follows the grouped runtime output directory.
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:tbb> "$<TARGET_FILE_DIR:tests>/"
  )
endif()

if (BUILD_AZURE_KINECT)
    # K4A headers are directly used in test. Currently we don't need to link
    # the K4A libraries.
    target_include_directories(tests SYSTEM PRIVATE ${K4A_INCLUDE_DIR})
endif()

if (BUILD_CUDA_MODULE)
    # We still need to explicitly link against CUDA libraries.
    # Consider removing dependencies of CUDA headers in the future.
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(tests PRIVATE CUDA::cudart)
endif()
