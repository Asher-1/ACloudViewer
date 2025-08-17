cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})

# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

project(COLMAP)

set(COLMAP_VERSION "${CLOUDVIEWER_VERSION}")
set(COLMAP_VERSION_NUMBER "${CLOUDVIEWER_VERSION_MAJOR}${CLOUDVIEWER_VERSION_MINOR}${CLOUDVIEWER_VERSION_PATCH}")

################################################################################
# Include CMake dependencies
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)

# Include helper macros and commands, and allow the included file to override
# the CMake policies in this file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)

################################################################################
# Options
################################################################################

option(BUILD_COLMAP_GUI "Builds COLMAP GUI" ON)
option(BOOST_STATIC "With boost static library" OFF)
option(CUDA_ENABLED "Whether to enable CUDA, if available" ON)
option(PBA_ENABLED "Whether to enable PBA for parallel BA Optimization" ON)
option(ALL_SOURCE_TARGET "Whether to create a target for all source files (for Visual Studio / XCode development)" OFF)
set(GUI_ENABLED ${BUILD_COLMAP_GUI})
set(OPENMP_ENABLED ${WITH_OPENMP})
set(TESTS_ENABLED ${BUILD_UNIT_TESTS})
set(CGAL_ENABLED ${CVCORELIB_USE_CGAL})
set(SIMD_ENABLED ${USE_SIMD})
set(OPENGL_ENABLED ON)
set(IPO_ENABLED ON)
set(PROFILING_ENABLED OFF) # Whether to enable google-perftools linker flags

if (UNIX OR APPLE)
    message(STATUS "set BOOST_STATIC: OFF on macos or linux")
    set(BOOST_STATIC OFF)
endif ()

if (TESTS_ENABLED)
    enable_testing()
endif ()

if (BOOST_STATIC)
    set(Boost_USE_STATIC ON)
    set(Boost_USE_MULTITHREAD)
    set(Boost_USE_STATIC_LIBS ON)
else ()
    # Force BOOST to use dynamic libraries (avoid link error with boost program_options)
    # https://lists.boost.org/boost-users/2009/11/54015.php
    add_compile_definitions(BOOST_ALL_DYN_LINK)
    add_compile_definitions(BOOST_TEST_DYN_LINK)
    set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_USE_STATIC OFF)
endif ()

# Build position-independent code, so that shared libraries can link against
# COLMAP's static libraries.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Find packages
################################################################################

if (OPENMP_ENABLED)
    find_package(OpenMP QUIET)
endif ()

# Linux: boost must building from source due to _GLIBCXX_USE_CXX11_ABI=0 issues
if (APPLE OR WIN32 OR ${GLIBCXX_USE_CXX11_ABI})
    find_package(Boost REQUIRED COMPONENTS
                graph
                filesystem
                program_options
                system)
    # find_package(FreeImage REQUIRED)
    # find_package(Glog REQUIRED)
    # find_package(Ceres REQUIRED)
    # find_package(Metis REQUIRED)
endif ()

if (NOT APPLE)
    if (WIN32)
        set(BOOST_ROOT "" CACHE PATH "Boost library root directory")
        if (NOT BOOST_ROOT)
            message(STATUS "No Boost library root specified (BOOST_ROOT)")
        else ()
            add_compile_definitions(BOOST_ROOT=${BOOST_ROOT})
            add_compile_definitions(BOOST_LIBRARYDIR=${BOOST_ROOT})
        endif ()
    else ()
        if (NOT ${GLIBCXX_USE_CXX11_ABI})
            set(BOOST_ROOT "${INTERNAL_BOOST_ROOT}" CACHE PATH "Boost library root directory")
        endif()
    endif ()

    message(STATUS "BOOST ROOT: ${BOOST_ROOT}")

    # Disable BOOST autolink
    if (MSVC)
        add_compile_definitions(BOOST_ALL_NO_LIB)
    endif ()

    if (BOOST_NO_CXX11)
        # Avoid link errors on boost filesystem copy_file function
        # http://stackoverflow.com/questions/35007134/c-boost-undefined-reference-to-boostfilesystemdetailcopy-file
        add_compile_definitions(BOOST_NO_CXX11_SCOPED_ENUMS)
        add_compile_definitions(BOOST_NO_CXX11_OVERRIDE)
        message("disable boost cxx11 on ${COLMAP_APP_NAME}!")
    endif ()
endif ()

set(OpenGL_GL_PREFERENCE GLVND)

if (CGAL_ENABLED)
    set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)
    # We do not use CGAL data. This prevents an unnecessary warning by CMake.
    set(CGAL_DATA_DIR "unused")
    find_package(CGAL QUIET)
endif ()

if (OPENGL_ENABLED AND NOT GUI_ENABLED)
    message(STATUS "Disabling GUI will also disable OpenGL")
    set(OPENGL_ENABLED OFF)
endif ()

if (CGAL_FOUND)
    list(APPEND CGAL_LIBRARY ${CGAL_LIBRARIES})
    message(STATUS "Found CGAL")
    message(STATUS "  Includes : ${CGAL_INCLUDE_DIRS}")
    message(STATUS "  Libraries : ${CGAL_LIBRARY}")
endif ()

################################################################################
# Compiler specific configuration
################################################################################

if (CMAKE_BUILD_TYPE)
    message(STATUS "Build type specified as ${CMAKE_BUILD_TYPE}")
else ()
    message(STATUS "Build type not specified, using Release")
    set(IS_DEBUG OFF)
endif ()

if (IS_MSVC)
    # Some fixes for the Glog library.
    add_compile_definitions(GLOG_NO_ABBREVIATED_SEVERITIES)
    add_compile_definitions(GOOGLE_GLOG_DLL_DECL=)
    # fix gflags library bugs
    add_compile_definitions(GFLAGS_DLL_DECLARE_FLAG= GFLAGS_DLL_DEFINE_FLAG= GFLAGS_IS_A_DLL=0)
    # opengl
    add_compile_definitions(GL_GLEXT_PROTOTYPES)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    # Enable object level parallel builds in Visual Studio.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif ()

if (IS_GNU)
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message(FATAL_ERROR "GCC version 4.8 or older not supported")
    endif ()

    # Hide incorrect warnings for uninitialized Eigen variables under GCC.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-maybe-uninitialized")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
endif ()

if (IS_DEBUG)
    add_compile_definitions(EIGEN_INITIALIZE_MATRICES_BY_NAN)
endif ()

if (OPENMP_ENABLED AND OPENMP_FOUND)
    message(STATUS "Enabling OpenMP support")
    add_compile_definitions(OPENMP_ENABLED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else ()
    message(STATUS "Disabling OpenMP support")
endif ()

if (IPO_ENABLED AND NOT IS_DEBUG AND NOT IS_GNU)
    message(STATUS "Enabling interprocedural optimization")
    set_property(DIRECTORY PROPERTY INTERPROCEDURAL_OPTIMIZATION 1)
else ()
    message(STATUS "Disabling interprocedural optimization")
endif ()

if (BUILD_CUDA_MODULE)
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_ARCHITECTURES}")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        message(
            FATAL_ERROR "You must set CMAKE_CUDA_ARCHITECTURES to e.g. 'native', 'all-major', '70', etc. "
            "More information at https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html")
    endif()

    set(CUDA_ENABLED ON)
    add_compile_definitions(CUDA_ENABLED)

    # Fix for some combinations of CUDA and GCC (e.g. under Ubuntu 16.04).
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES")

    # Do not show warnings if the architectures are deprecated.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

    message(STATUS "PBA enabled - CUDA ${CUDAToolkit_VERSION} supports modern texture API")
    # Check CUDA version for PBA compatibility
    # if(CUDAToolkit_VERSION VERSION_LESS "12.0")
    #     set(PBA_ENABLED ON)
    #     message(STATUS "PBA enabled - CUDA ${CUDAToolkit_VERSION} supports modern texture API")
    # else()
    #     set(PBA_ENABLED OFF)
    #     message(STATUS "PBA disabled - requires CUDA 11.x or below (but found CUDA ${CUDAToolkit_VERSION})")
    # endif()

    message(STATUS "Enabling CUDA support (version: ${CUDAToolkit_VERSION},"
            " archs: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(CUDA_ENABLED OFF)
    message(STATUS "Disabling CUDA support")
endif ()
message(STATUS "Colmap CUDA enabled: ${CUDA_ENABLED}")

if (GUI_ENABLED AND Qt5_FOUND)
    add_compile_definitions(GUI_ENABLED)
    message(STATUS "Enabling GUI support")
else ()
    message(STATUS "Disabling GUI support")
endif ()

if (OPENGL_ENABLED)
    add_compile_definitions(OPENGL_ENABLED)
    message(STATUS "Enabling OpenGL support")
else ()
    message(STATUS "Disabling OpenGL support")
endif ()

if (PROFILING_ENABLED)
    message(STATUS "Enabling profiling support")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lprofiler -ltcmalloc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lprofiler -ltcmalloc")
else ()
    message(STATUS "Disabling profiling support")
endif ()

if (CGAL_FOUND AND CGAL_ENABLED)
    message(STATUS "Enabling CGAL support")
    add_compile_definitions(CGAL_ENABLED)
else ()
    message(STATUS "Disabling CGAL support")
    set(CGAL_ENABLED OFF)
endif ()

if (Qt5_FOUND)
    # Qt5 was built with -reduce-relocations.
    if (Qt5_POSITION_INDEPENDENT_CODE)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        if (CUDA_ENABLED AND NOT IS_MSVC)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}--compiler-options=-fPIC")
        endif ()
    endif ()

    # Enable automatic compilation of Qt resource files.
    set(CMAKE_AUTORCC ON)
endif ()

################################################################################
# Add sources
################################################################################

# Generate source file with version definitions.
include(GenerateVersionDefinitions)
set(COLMAP_INCLUDE_DIRS
        ${Boost_INCLUDE_DIR}
        ${EIGEN_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${FREEIMAGE_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/include
        )

set(COLMAP_LINK_DIRS
        ${Boost_LIBRARY_DIRS}
        )

if (WIN32) 
    set(COLMAP_EXTERNAL_LIBRARIES
        ${CMAKE_DL_LIBS}
        Boost::boost
        Boost::graph
        Boost::system
        Boost::filesystem
        Boost::program_options
        )
else()
    set(COLMAP_EXTERNAL_LIBRARIES
        ${CMAKE_DL_LIBS}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        # ${GLOG_LIBRARIES}
        # ${CERES_LIBRARIES}
        # ${FREEIMAGE_LIBRARIES}
        )
endif()
        
if (OPENMP_FOUND)
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${OpenMP_libomp_LIBRARY})
endif ()

if (Qt5_FOUND AND GUI_ENABLED)
    list(APPEND COLMAP_INCLUDE_DIRS ${Qt5Core_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS})
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${Qt5Core_LIBRARIES} ${Qt5OpenGL_LIBRARIES} ${Qt5Widgets_LIBRARIES})
endif ()

if (CGAL_FOUND)
    list(APPEND COLMAP_INCLUDE_DIRS ${CGAL_INCLUDE_DIRS} ${GMP_INCLUDE_DIR})
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${CGAL_LIBRARY} ${GMP_LIBRARIES})
    list(APPEND COLMAP_LINK_DIRS ${CGAL_LIBRARIES_DIR})
endif ()

if (UNIX)
    list(APPEND COLMAP_EXTERNAL_LIBRARIES pthread)
endif ()

set(COLMAP_INTERNAL_LIBRARIES
        flann
        graclus
        lsd
        )

# Add PBA conditionally based on CUDA version
if(PBA_ENABLED)
    list(APPEND COLMAP_INTERNAL_LIBRARIES pba)
    add_compile_definitions(PBA_ENABLED)
else()
    message(STATUS "PBA library excluded from build")
endif()

list(APPEND COLMAP_INTERNAL_LIBRARIES
        poisson_recon
        sqlite3
        sift_gpu
        vlfeat
        )

include_directories(
        lib
        src
        ${COLMAP_INCLUDE_DIRS}
)

link_directories(${COLMAP_LINK_DIRS})

add_subdirectory(lib)
add_subdirectory(src)

################################################################################
# Generate source groups for Visual Studio, XCode, etc.
################################################################################

COLMAP_ADD_SOURCE_DIR(lib/FLANN LIB_FLANN_SRCS *.h *.cpp *.hpp *.cu)
COLMAP_ADD_SOURCE_DIR(lib/Graclus LIB_GRACLUS_SRCS *.h *.c)
COLMAP_ADD_SOURCE_DIR(lib/LSD LIB_LSD_SRCS *.h *.c)

COLMAP_ADD_SOURCE_DIR(lib/PBA LIB_PBA_SRCS *.h *.cpp *.cu)
COLMAP_ADD_SOURCE_DIR(lib/SiftGPU LIB_SIFT_GPU_SRCS *.h *.cpp *.cu)
COLMAP_ADD_SOURCE_DIR(lib/PoissonRecon LIB_POISSON_RECON_SRCS *.h *.cpp *.inl)
COLMAP_ADD_SOURCE_DIR(lib/SQLite LIB_SQLITE_SRCS *.h *.c)
COLMAP_ADD_SOURCE_DIR(lib/VLFeat LIB_VLFEAT_SRCS *.h *.c *.tc)

COLMAP_ADD_SOURCE_DIR(src/base BASE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/controllers CONTROLLERS_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/estimators ESTIMATORS_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/exe EXE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/feature FEATURE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/mvs MVS_SRCS *.h *.cc *.cu)
COLMAP_ADD_SOURCE_DIR(src/optim OPTIM_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/retrieval RETRIEVAL_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/sfm SFM_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/tools TOOLS_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/ui UI_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/util UTIL_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/pipelines PIPELINES_SRCS *.h *.cpp *.cc *.hpp)

# Add all of the source files to a regular library target, as using a custom
# target does not allow us to set its C++ include directories (and thus
# intellisense can't find any of the included files).
if(ALL_SOURCE_TARGET)
    add_library(${COLMAP_SRC_ROOT_FOLDER}
            ${LIB_FLANN_SRCS}
            ${LIB_GRACLUS_SRCS}
            ${LIB_LSD_SRCS}
            ${LIB_PBA_SRCS}
            ${LIB_POISSON_RECON_SRCS}
            ${LIB_SIFT_GPU_SRCS}
            ${LIB_SQLITE_SRCS}
            ${LIB_VLFEAT_SRCS}
            ${BASE_SRCS}
            ${CONTROLLERS_SRCS}
            ${ESTIMATORS_SRCS}
            ${EXE_SRCS}
            ${FEATURE_SRCS}
            ${MVS_SRCS}
            ${OPTIM_SRCS}
            ${RETRIEVAL_SRCS}
            ${SFM_SRCS}
            ${TOOLS_SRCS}
            ${UI_SRCS}
            ${UTIL_SRCS}
            ${PIPELINES_SRCS}
            )

    # Prevent the library from being compiled automatically.
    set_target_properties(
            ${COLMAP_SRC_ROOT_FOLDER} PROPERTIES
            EXCLUDE_FROM_ALL 1
            EXCLUDE_FROM_DEFAULT_BUILD 1)
endif()

################################################################################
# Install and uninstall scripts
################################################################################

# Install header files.
install(DIRECTORY src/
        DESTINATION include/${COLMAP_APP_NAME}
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY lib/
        DESTINATION include/${COLMAP_APP_NAME}/lib
        FILES_MATCHING REGEX ".*[.]h|.*[.]hpp|.*[.]inl")

# Generate and install CMake configuration.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/COLMAPConfig.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/COLMAPConfig.cmake"
        DESTINATION "share/${COLMAP_APP_NAME}")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeConfigVersion.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/COLMAPConfigVersion.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/COLMAPConfigVersion.cmake"
        DESTINATION "share/${COLMAP_APP_NAME}")

# Install find_package scripts for dependencies.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake
        DESTINATION share/${COLMAP_APP_NAME}
        FILES_MATCHING PATTERN "Find*.cmake")

# Install batch scripts under Windows.
if (IS_MSVC)
    install(FILES "${COLOUDVIEWER_PROJECT_PATH}/scripts/reconstruction/shell/${COLMAP_APP_NAME}.bat"
            "${COLOUDVIEWER_PROJECT_PATH}/scripts/reconstruction/shell/RUN_TESTS.bat"
            DESTINATION "/"  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif ()

# Configure the uninstallation script.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeUninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake" IMMEDIATE @ONLY)
add_custom_target(uninstall_colmap COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake)
set_target_properties(uninstall_colmap PROPERTIES FOLDER ${CMAKE_TARGETS_ROOT_FOLDER})
