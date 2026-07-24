# AICore: Unified AI inference core library
# Capability modules (not plugin names):
#   src/depth/    — multi-view monocular depth (DA3-family GGUF)
#   src/gaussian/ — image-to-3D Gaussian splatting (FreeSplatter-family GGUF)
#   src/lightglue/ — sparse feature matching (LightGlue GGUF)
#   src/common/   — shared utilities (data paths, ggml helpers, logging)
#
# Build:   libAICore.so
# Exports: all public headers under include/aicore/
# Depends: 3rdparty_ggml (PRIVATE), CVCoreLib, Qt::Core + Qt::Gui

if(NOT AICore_ENABLED)
    message(STATUS "AICore module is disabled (set AICore_ENABLED=ON to enable)")
    return()
endif()

if(NOT TARGET 3rdparty_ggml)
    message(WARNING "AICore: 3rdparty_ggml not available. Enable AICore_ENABLED in CMake.")
    set(AICore_ENABLED OFF CACHE BOOL "" FORCE)
    return()
endif()

project(AICore
    DESCRIPTION "Unified AI inference core — depth + gaussian + lightglue"
    LANGUAGES CXX
)

set(AICORE_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(AICORE_COMMON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/common)
set(AICORE_DEPTH_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/depth)
set(AICORE_GAUSSIAN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian)
set(AICORE_LIGHTGLUE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/lightglue)

if(NOT EXISTS ${AICORE_DEPTH_SRC_DIR}/engine.cpp)
    message(WARNING "AICore: depth module not found at ${AICORE_DEPTH_SRC_DIR}")
    set(AICore_ENABLED OFF CACHE BOOL "" FORCE)
    return()
endif()

if(NOT EXISTS ${AICORE_GAUSSIAN_SRC_DIR}/model.cpp)
    message(WARNING "AICore: gaussian module not found at ${AICORE_GAUSSIAN_SRC_DIR}")
    set(AICore_ENABLED OFF CACHE BOOL "" FORCE)
    return()
endif()

if(NOT EXISTS ${AICORE_LIGHTGLUE_SRC_DIR}/matcher.cpp)
    message(WARNING "AICore: lightglue module not found at ${AICORE_LIGHTGLUE_SRC_DIR}")
    set(AICore_ENABLED OFF CACHE BOOL "" FORCE)
    return()
endif()

file(GLOB AICORE_COMMON_SOURCES ${AICORE_COMMON_SRC_DIR}/*.cpp)
file(GLOB AICORE_COMMON_HEADERS ${AICORE_COMMON_SRC_DIR}/*.hpp)
file(GLOB DEPTH_SOURCES ${AICORE_DEPTH_SRC_DIR}/*.cpp)
file(GLOB DEPTH_HEADERS ${AICORE_DEPTH_SRC_DIR}/*.hpp)
file(GLOB GAUSSIAN_SOURCES ${AICORE_GAUSSIAN_SRC_DIR}/*.cpp)
file(GLOB GAUSSIAN_HEADERS ${AICORE_GAUSSIAN_SRC_DIR}/*.hpp ${AICORE_GAUSSIAN_SRC_DIR}/*.h)
file(GLOB LIGHTGLUE_SOURCES ${AICORE_LIGHTGLUE_SRC_DIR}/*.cpp)
file(GLOB LIGHTGLUE_HEADERS ${AICORE_LIGHTGLUE_SRC_DIR}/*.hpp)
file(GLOB AICORE_API_HEADERS ${AICORE_INC_DIR}/aicore/*.h)

add_library(${PROJECT_NAME} SHARED
    ${AICORE_COMMON_SOURCES} ${AICORE_COMMON_HEADERS}
    ${DEPTH_SOURCES} ${DEPTH_HEADERS}
    ${GAUSSIAN_SOURCES} ${GAUSSIAN_HEADERS}
    ${LIGHTGLUE_SOURCES} ${LIGHTGLUE_HEADERS}
    ${AICORE_API_HEADERS}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    FOLDER "core"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# AICore and its private ggml runtime files are always deployed together. Use
# only a loader-relative runtime path so build-machine directories cannot leak
# into installers or wheels.
if(APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "$ORIGIN")
endif()

if(UNIX AND NOT APPLE)
    target_link_options(${PROJECT_NAME} PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/aicore.exports.map")
    set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
        LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/aicore.exports.map)
endif()

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${AICORE_INC_DIR}>
        $<INSTALL_INTERFACE:include/AICore>
    PRIVATE
        ${AICORE_COMMON_SRC_DIR}
        ${AICORE_DEPTH_SRC_DIR}
        ${AICORE_GAUSSIAN_SRC_DIR}
        ${AICORE_LIGHTGLUE_SRC_DIR}
)

add_dependencies(${PROJECT_NAME} 3rdparty_ggml)
target_link_libraries(${PROJECT_NAME} PRIVATE 3rdparty_ggml)
if(DEFINED GGML_INCLUDE_DIRS)
    target_include_directories(${PROJECT_NAME} PRIVATE ${GGML_INCLUDE_DIRS})
endif()

# Dynamic backend mode: GPU backends are separate modules loaded at runtime.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AICoreCompileDefinitions.cmake)
if(GGML_DYNAMIC_BACKENDS)
    message(STATUS "AICore: using dynamic ggml backends (AICORE_BACKEND_DL)")
endif()
aicore_apply_compile_definitions(${PROJECT_NAME})

# Static mode only: ggml-cuda static archives are embedded into libAICore.so
# and pull in CUDA driver API (libcuda.so) as a DT_NEEDED dep.  In Docker/CI
# environments the real driver isn't installed; the CUDA toolkit ships a stub at
# <toolkit>/lib64/stubs/.  Expose that path so downstream executables (colmap_exe,
# ACloudViewer) can resolve libcuda.so at link time.
if(NOT GGML_DYNAMIC_BACKENDS AND AICore_CUDA_ENABLED AND CUDAToolkit_FOUND)
    get_target_property(_cuda_driver_loc CUDA::cuda_driver IMPORTED_LOCATION)
    if(_cuda_driver_loc)
        get_filename_component(_cuda_driver_dir "${_cuda_driver_loc}" DIRECTORY)
        target_link_directories(${PROJECT_NAME} PUBLIC "${_cuda_driver_dir}")
    elseif(CUDAToolkit_LIBRARY_DIR)
        if(EXISTS "${CUDAToolkit_LIBRARY_DIR}/stubs")
            target_link_directories(${PROJECT_NAME} PUBLIC "${CUDAToolkit_LIBRARY_DIR}/stubs")
        endif()
    endif()
endif()

if(TARGET CVCoreLib)
    target_link_libraries(${PROJECT_NAME} PRIVATE CVCoreLib)
    target_compile_definitions(${PROJECT_NAME} PRIVATE AICore_HAS_CVLOG)
else()
    message(FATAL_ERROR "AICore requires CVCoreLib (CVTools for cross-platform paths)")
endif()

if(TARGET Qt::Core AND TARGET Qt::Gui)
    target_link_libraries(${PROJECT_NAME} PUBLIC Qt::Core Qt::Gui)
elseif(Qt5_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Qt5::Core Qt5::Gui)
endif()

target_compile_definitions(${PROJECT_NAME}
    PRIVATE AICore_BUILD AICore_SHARED
    INTERFACE AICore_SHARED
    PUBLIC AICore_ENABLED
)
if(MSVC)
    # M_PI and other math constants (pose.cpp, tests); same as CloudViewerSetGlobalProperties.cmake
    target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)
endif()

include(CheckCXXCompilerFlag)
if(ENABLE_AVX512)
    if(NOT MSVC)
        check_cxx_compiler_flag("-mavx512f" AICORE_HAS_AVX512F)
        if(AICORE_HAS_AVX512F)
            set_source_files_properties(${AICORE_DEPTH_SRC_DIR}/winograd.cpp
                PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma")
        endif()
    endif()
else()
    if(NOT MSVC)
        check_cxx_compiler_flag("-mavx2" AICORE_HAS_AVX2)
        check_cxx_compiler_flag("-mfma" AICORE_HAS_FMA)
        if(AICORE_HAS_AVX2 AND AICORE_HAS_FMA)
            set_source_files_properties(${AICORE_DEPTH_SRC_DIR}/winograd.cpp
                PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
        endif()
    endif()
endif()

set(AICore_FOUND TRUE CACHE BOOL "AICore module found" FORCE)
set(AICore_INCLUDE_DIRS
    "${AICORE_INC_DIR}"
    CACHE PATH "AICore include dirs" FORCE)
set(AICore_LIBRARIES AICore CACHE STRING "AICore library target" FORCE)

if(NOT APPLE)
    cloudViewer_install_targets(${PROJECT_NAME})
    InstallSharedLibrary(TARGET ${PROJECT_NAME})
endif()
install(DIRECTORY ${AICORE_INC_DIR}/aicore
        DESTINATION include/AICore
        FILES_MATCHING PATTERN "*.h")

# Ship ggml shared libs and dynamic backend modules alongside libAICore.
# file(GLOB) runs at configure time before ExternalProject builds, so we use a
# POST_BUILD script that globs at build time when the dylibs exist.
#
# IMPORTANT: In GGML_BACKEND_DL mode, ggml installs dynamic backend modules as
# .so files (MODULE libraries) into the bin/ subdirectory of the install prefix,
# even on macOS.  ggml's runtime loader (ggml-backend-reg.cpp) searches for .so
# extensions on all UNIX platforms.  We must copy both:
#   - .dylib from lib/  (shared core: ggml, ggml-base)
#   - .so    from bin/  (loadable modules for runtime: ggml-metal, ggml-vulkan, etc.)
if(GGML_DYNAMIC_BACKENDS AND GGML_RUNTIME_LIB_DIR AND GGML_BACKEND_DIR)
    string(REPLACE ";" "|" _aicore_gpu_backend_files_pipe
        "${GGML_GPU_BACKEND_MODULE_FILES}")
    set(_aicore_copy_ggml_script
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CopyGgmlBackends.cmake")
    set(_aicore_backend_copy_args
        -DCOPY_BACKEND_MODULES=ON
        -DGGML_BACKEND_DIR="${GGML_BACKEND_DIR}"
        -DDEST_DIR="$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        -DLIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        -DLIB_SUFFIX="${GGML_MODULE_SUFFIX}"
        -DGPU_FILES_PIPE="${_aicore_gpu_backend_files_pipe}")
    if(GGML_CPU_ALL_VARIANTS)
        list(APPEND _aicore_backend_copy_args -DCPU_ALL_VARIANTS=ON)
    endif()
    list(APPEND _aicore_backend_copy_args -P "${_aicore_copy_ggml_script}")
    # Copy shared libraries (.dylib / .so depending on platform)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DCOPY_CORE_SHARED=ON
            -DGGML_BACKEND_DIR="${GGML_RUNTIME_LIB_DIR}"
            -DDEST_DIR="$<TARGET_FILE_DIR:${PROJECT_NAME}>"
            -DLIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
            -DLIB_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}"
            -P "${_aicore_copy_ggml_script}"
        COMMENT "Copying ggml shared libraries to libAICore directory"
    )
    # Copy loadable backend modules (.so on all UNIX platforms, .dll on Windows)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} ${_aicore_backend_copy_args}
        COMMENT "Copying ggml dynamic backend modules to libAICore directory"
    )

    # `cmake --install` only ships libAICore.so by default; backend MODULE libs
    # (libggml-cpu.so, libggml-cuda.so, …) must be installed alongside it or
    # ggml_backend_load_all_from_path() finds dev_count=0 at runtime.
    if(POLICY CMP0087)
        cmake_policy(SET CMP0087 NEW)
    endif()
    if(WIN32)
        set(_aicore_ggml_install_subdir "${ACloudViewer_DEST_FOLDER}")
    else()
        set(_aicore_ggml_install_subdir "${CloudViewer_INSTALL_LIB_DIR}")
    endif()
    if(GGML_CPU_ALL_VARIANTS)
        set(_aicore_install_cpu_all_variants ON)
    else()
        set(_aicore_install_cpu_all_variants OFF)
    endif()
    set(_AICORE_GPU_PIPE "${_aicore_gpu_backend_files_pipe}")
    set(_AICORE_CPU_ALL_VARIANTS "${_aicore_install_cpu_all_variants}")
    set(_AICORE_COPY_GGML_SCRIPT "${_aicore_copy_ggml_script}")
    set(_AICORE_GGML_INSTALL_SUBDIR "${_aicore_ggml_install_subdir}")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/InstallGgmlBackends.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/InstallGgmlBackends.cmake"
        @ONLY)
    install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/InstallGgmlBackends.cmake")
endif()

if(AICore_BUILD_TESTS)
    if(AICore_BUILD_WHITEBOX_TESTS)
        set(AICORE_TEST_SOURCES
            ${AICORE_COMMON_SOURCES}
            ${DEPTH_SOURCES}
            ${GAUSSIAN_SOURCES}
            ${LIGHTGLUE_SOURCES}
        )
        list(FILTER AICORE_TEST_SOURCES EXCLUDE REGEX
             ".*/(capi|image_depth)\\.cpp$")

        add_library(AICore_test STATIC
            ${AICORE_TEST_SOURCES}
            ${AICORE_COMMON_HEADERS} ${DEPTH_HEADERS} ${GAUSSIAN_HEADERS}
            ${LIGHTGLUE_HEADERS}
        )
        set_target_properties(AICore_test PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            FOLDER "core/tests"
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )
        target_include_directories(AICore_test PUBLIC
            $<BUILD_INTERFACE:${AICORE_INC_DIR}>
            $<BUILD_INTERFACE:${AICORE_COMMON_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_DEPTH_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_GAUSSIAN_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_LIGHTGLUE_SRC_DIR}>
        )
        add_dependencies(AICore_test 3rdparty_ggml)
        target_link_libraries(AICore_test PUBLIC 3rdparty_ggml CVCoreLib)
        if(DEFINED GGML_INCLUDE_DIRS)
            target_include_directories(AICore_test PUBLIC ${GGML_INCLUDE_DIRS})
        endif()
        if(TARGET Qt::Core AND TARGET Qt::Gui)
            target_link_libraries(AICore_test PUBLIC Qt::Core Qt::Gui)
        elseif(Qt5_FOUND)
            target_link_libraries(AICore_test PUBLIC Qt5::Core Qt5::Gui)
        endif()
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(AICore_test PUBLIC OpenMP::OpenMP_CXX)
        endif()
        target_compile_definitions(AICore_test PUBLIC AICore_ENABLED)
        aicore_apply_compile_definitions(AICore_test)
        if(MSVC)
            target_compile_definitions(AICore_test PRIVATE _USE_MATH_DEFINES)
        endif()
    endif()

    add_subdirectory(tests)
    message(STATUS "AICore: contract tests enabled")
endif()

message(STATUS "AICore (SHARED): depth + gaussian + lightglue modules, ggml linked once")
