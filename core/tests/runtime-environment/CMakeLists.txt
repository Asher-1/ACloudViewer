cmake_minimum_required(VERSION 3.24)
project(CloudViewerRuntimeEnvironmentTests LANGUAGES C)

if(NOT APPLE)
    message(FATAL_ERROR "These tests exercise the macOS dynamic loader")
endif()

enable_testing()
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/CloudViewerOutputDirectories.cmake")

# An explicit dependency must survive a same-named library in the test path.
add_library(pinned SHARED dependency.c)
target_compile_definitions(pinned PRIVATE DEPENDENCY_VALUE=1)
set_target_properties(pinned PROPERTIES
    OUTPUT_NAME cv_env_probe
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pinned"
    BUILD_WITH_INSTALL_NAME_DIR TRUE
    INSTALL_NAME_DIR "${CMAKE_BINARY_DIR}/pinned"
    MACOSX_RPATH OFF)
add_library(shadow SHARED dependency.c)
target_compile_definitions(shadow PRIVATE DEPENDENCY_VALUE=2)
set_target_properties(shadow PROPERTIES
    OUTPUT_NAME cv_env_probe
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_executable(pinned_probe probe.c)
target_compile_definitions(pinned_probe PRIVATE EXPECTED_VALUE=1)
target_link_libraries(pinned_probe PRIVATE pinned)
add_test(NAME preserve_pinned_dependency COMMAND pinned_probe)
cloudviewer_set_test_runtime_environment(preserve_pinned_dependency)

# Bare install names still need the build-tree fallback when no RPATH applies.
add_library(fallback SHARED dependency.c)
target_compile_definitions(fallback PRIVATE DEPENDENCY_VALUE=3)
set_target_properties(fallback PROPERTIES
    OUTPUT_NAME cv_fallback_probe
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    BUILD_WITH_INSTALL_NAME_DIR TRUE
    INSTALL_NAME_DIR ""
    MACOSX_RPATH OFF)
add_executable(fallback_probe probe.c)
target_compile_definitions(fallback_probe PRIVATE EXPECTED_VALUE=3)
target_link_libraries(fallback_probe PRIVATE fallback)
set_target_properties(fallback_probe PROPERTIES SKIP_BUILD_RPATH TRUE)
add_test(NAME resolve_build_dependency COMMAND fallback_probe)
cloudviewer_set_test_runtime_environment(resolve_build_dependency)
