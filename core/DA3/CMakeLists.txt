# DA3 (Depth Anything V3) algorithm module
# Part of CVCoreLib family — provides monocular depth estimation, camera pose
# inference, 3D reconstruction, and COLMAP-compatible export.
#
# Builds: DA3Core (static library)
# Depends: 3rdparty_ggml (required), Qt::Core + Qt::Gui (image I/O)
# Consumers: CV_DB_LIB, ColmapLib, qDA3 plugin, etc.

# DA3_ENABLED option is defined in the root CMakeLists.txt
if(NOT DA3_ENABLED)
    message(STATUS "DA3 module is disabled (set DA3_ENABLED=ON to enable)")
    return()
endif()

# ggml should be available from find_dependencies.cmake
if(NOT TARGET 3rdparty_ggml)
    message(WARNING "DA3: 3rdparty_ggml not available. Enable GGML_ENABLED in CMake.")
    set(DA3_ENABLED OFF CACHE BOOL "" FORCE)
    return()
endif()

project(DA3Core
    DESCRIPTION "Depth Anything V3 algorithm core — monocular depth, pose, and 3D reconstruction"
    LANGUAGES CXX
)

set(DA3_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(DA3_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(NOT EXISTS ${DA3_SRC_DIR}/engine.cpp)
    message(WARNING "DA3: source files not found at ${DA3_SRC_DIR}")
    set(DA3_ENABLED OFF CACHE BOOL "" FORCE)
    return()
endif()

file(GLOB DA3_SOURCES ${DA3_SRC_DIR}/*.cpp)
file(GLOB DA3_HEADERS ${DA3_SRC_DIR}/*.hpp ${DA3_INC_DIR}/*.h)

add_library(${PROJECT_NAME} STATIC ${DA3_SOURCES} ${DA3_HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    FOLDER "core"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${DA3_INC_DIR}>
        $<BUILD_INTERFACE:${DA3_SRC_DIR}>
        $<INSTALL_INTERFACE:include/DA3>
)

# ggml inference backend (PUBLIC: consumers that include DA3 internal headers
# such as engine.hpp/model_loader.hpp need ggml include paths transitively)
add_dependencies(${PROJECT_NAME} 3rdparty_ggml)
target_link_libraries(${PROJECT_NAME} PUBLIC 3rdparty_ggml)

# CVCoreLib for CVLog integration and CVTools (path/string helpers)
if(TARGET CVCoreLib)
    target_link_libraries(${PROJECT_NAME} PRIVATE CVCoreLib)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DA3_HAS_CVLOG)
else()
    message(FATAL_ERROR "DA3Core requires CVCoreLib (CVTools for cross-platform paths)")
endif()

# Qt image I/O (replaces stb)
if(TARGET Qt::Core AND TARGET Qt::Gui)
    target_link_libraries(${PROJECT_NAME} PRIVATE Qt::Core Qt::Gui)
elseif(Qt5_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE Qt5::Core Qt5::Gui)
endif()

# SIMD optimization for winograd — tiered: AVX-512 > AVX2+FMA > scalar
#
# AVX-512: opt-in via ENABLE_AVX512 (default OFF). Only for CPUs that support it.
# AVX2+FMA: auto-detected. Covers most x86_64 CPUs since 2013 (Haswell / Zen1).
# Scalar: always-available fallback, ~4x slower than AVX2.
include(CheckCXXCompilerFlag)
if(ENABLE_AVX512)
    if(MSVC)
        check_cxx_compiler_flag("/arch:AVX512" DA3_HAS_AVX512F)
        if(DA3_HAS_AVX512F)
            set_source_files_properties(
                ${DA3_SRC_DIR}/winograd.cpp
                PROPERTIES COMPILE_OPTIONS "/arch:AVX512"
            )
            message(STATUS "DA3 winograd: AVX-512 (MSVC)")
        endif()
    else()
        check_cxx_compiler_flag("-mavx512f" DA3_HAS_AVX512F)
        if(DA3_HAS_AVX512F)
            set_source_files_properties(
                ${DA3_SRC_DIR}/winograd.cpp
                PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma"
            )
            message(STATUS "DA3 winograd: AVX-512 (GCC/Clang)")
        endif()
    endif()
else()
    if(MSVC)
        check_cxx_compiler_flag("/arch:AVX2" DA3_HAS_AVX2)
        if(DA3_HAS_AVX2)
            set_source_files_properties(
                ${DA3_SRC_DIR}/winograd.cpp
                PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
            )
            message(STATUS "DA3 winograd: AVX2+FMA (MSVC)")
        else()
            message(STATUS "DA3 winograd: scalar fallback")
        endif()
    else()
        check_cxx_compiler_flag("-mavx2" DA3_HAS_AVX2)
        check_cxx_compiler_flag("-mfma" DA3_HAS_FMA)
        if(DA3_HAS_AVX2 AND DA3_HAS_FMA)
            set_source_files_properties(
                ${DA3_SRC_DIR}/winograd.cpp
                PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma"
            )
            message(STATUS "DA3 winograd: AVX2+FMA (GCC/Clang)")
        else()
            message(STATUS "DA3 winograd: scalar fallback")
        endif()
    endif()
endif()

# Compile definition for conditional compilation in consumers
target_compile_definitions(${PROJECT_NAME} PUBLIC DA3_ENABLED)

if(DA3_ENABLED)
    if(APPLE)
        message(STATUS "DA3Core: GPU inference via ggml Metal (macOS)")
    elseif(BUILD_CUDA_MODULE OR GGML_USE_CUDA)
        message(STATUS "DA3Core: GPU inference via ggml CUDA")
    elseif(GGML_USE_VULKAN)
        message(STATUS "DA3Core: GPU inference via ggml Vulkan")
    else()
        message(STATUS "DA3Core: CPU-only ggml (enable BUILD_CUDA_MODULE or GGML_USE_CUDA on Windows/Linux for GPU)")
    endif()
endif()

# Export variables for downstream consumers
set(DA3_FOUND TRUE CACHE BOOL "DA3 core module found" FORCE)
set(DA3_INCLUDE_DIRS "${DA3_INC_DIR};${DA3_SRC_DIR}" CACHE PATH "DA3 include dirs" FORCE)
set(DA3_LIBRARIES DA3Core CACHE STRING "DA3 library target" FORCE)

message(STATUS "DA3Core: ${DA3_SRC_DIR} (${CMAKE_CURRENT_SOURCE_DIR})")
