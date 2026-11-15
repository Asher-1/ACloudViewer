cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})

# Suppress CMP0167 warning
if (POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

project(COLMAP)

# Upstream parity (d3ccaf35): upstream compiles COLMAP with compiler-default
# codegen. The fork-wide PCL-compatibility macro EIGEN_MAX_ALIGN_BYTES=0 and
# the SIMD flags (-msse4.2 -mfpmath=sse -mno-avx512*) change Eigen's codegen
# inside the COLMAP subtree: the shared-focal RANSAC residual evaluation
# (RelativePoseSharedFocalEstimator::Residuals, estimators/solvers/*.cc)
# flips best-model selection at threshold ties, which chaotically diverges
# trial counts and two-view-geometry inlier sets (observed: TVG pair (4,5)
# inlier sets differ by one match; SF focal 3296.79 vs 3293.38 px). Strip
# both for this subproject so every COLMAP TU compiles like upstream. Eigen
# objects in this subtree are function-local, so the codegen change cannot
# leak across translation units (no ODR/layout exposure).
get_directory_property(COLMAP_DIR_DEFS COMPILE_DEFINITIONS)
list(REMOVE_ITEM COLMAP_DIR_DEFS "EIGEN_MAX_ALIGN_BYTES=0")
set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS "${COLMAP_DIR_DEFS}")
string(REPLACE " -msse4.2" "" COLMAP_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -mfpmath=sse" "" COLMAP_CXX_FLAGS "${COLMAP_CXX_FLAGS}")
string(REPLACE " -mno-avx512f" "" COLMAP_CXX_FLAGS "${COLMAP_CXX_FLAGS}")
string(REPLACE " -mno-avx512cd" "" COLMAP_CXX_FLAGS "${COLMAP_CXX_FLAGS}")
string(REPLACE " -mno-avx512vl" "" COLMAP_CXX_FLAGS "${COLMAP_CXX_FLAGS}")
string(REPLACE " -mno-avx512bw" "" COLMAP_CXX_FLAGS "${COLMAP_CXX_FLAGS}")
string(REPLACE " -mno-avx512dq" "" COLMAP_CXX_FLAGS "${COLMAP_CXX_FLAGS}")
string(REPLACE " -mno-avx512ifma" "" COLMAP_CXX_FLAGS "${COLMAP_CXX_FLAGS}")
string(REPLACE " -mno-avx512pf" "" COLMAP_CXX_FLAGS "${COLMAP_CXX_FLAGS}")
string(REPLACE " -mno-avx512er" "" COLMAP_CXX_FLAGS "${COLMAP_CXX_FLAGS}")
string(REPLACE " -mno-avx512vbmi2" "" COLMAP_CXX_FLAGS "${COLMAP_CXX_FLAGS}")
string(REPLACE " -mno-avx512vbmi" "" COLMAP_CXX_FLAGS "${COLMAP_CXX_FLAGS}")
string(REPLACE " -mno-avx512vnni" "" COLMAP_CXX_FLAGS "${COLMAP_CXX_FLAGS}")
string(REPLACE " -mno-avx512bitalg" "" COLMAP_CXX_FLAGS "${COLMAP_CXX_FLAGS}")
string(REPLACE " -mno-avx512vpopcntdq" "" COLMAP_CXX_FLAGS "${COLMAP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${COLMAP_CXX_FLAGS}")

# Hash containers: the scene/SfM containers use the FlatHashMap/NodeHashMap
# aliases from util/hash_containers.h, backed by the vendored boost::unordered
# 1.92 headers exactly as upstream (see src/CMakeLists.txt). The earlier std
# backend was measured against the upstream 3e5c915c executable on the L3
# mini6 fixture: keypoints, matches, and two-view geometries were identical,
# yet the models diverged (1241 vs 1237 track observations), and upstream
# rebuilt with std containers reproduced the fork's model to 1e-15. The
# iteration order of these containers is therefore observable SfM behavior,
# not an implementation detail.


# The COLMAP engine version this inlined fork is aligned with. It drives the
# database schema version (see src/util/version.h) and is decoupled from the
# CLOUDVIEWER package version on purpose.
set(COLMAP_VERSION "4.3.0.dev0")
set(COLMAP_VERSION_MAJOR "4")
set(COLMAP_VERSION_MINOR "3")
set(COLMAP_VERSION_PATCH "0")

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
option(RECONSTRUCTION_HIP_ENABLED
       "Build COLMAP PatchMatch with the ROCm HIP toolchain" OFF)
option(PBA_ENABLED "Whether to enable PBA for parallel BA Optimization" OFF) # deprecated: PBA backend retired with the COLMAP alignment (c9729fa0)
option(RECONSTRUCTION_FETCH_POSELIB
       "Fetch the pinned PoseLib minimal solvers for reconstruction" ON)
option(RECONSTRUCTION_CASPAR_ENABLED
       "Enable COLMAP Caspar CUDA bundle adjustment" OFF)
option(RECONSTRUCTION_CASPAR_USE_DOUBLE
       "Use f64 generated Caspar kernels" OFF)
option(RECONSTRUCTION_L3_GATE_ENABLED
       "Enable the fixed-fixture candidate-versus-pinned-upstream L3 SfM gate" OFF)
set(RECONSTRUCTION_L3_IMAGES "" CACHE PATH
    "Fixed image fixture directory for reconstruction_l3_alignment_gate")
set(RECONSTRUCTION_L3_ACV_COLMAP "" CACHE FILEPATH
    "Candidate ACloudViewer COLMAP executable for reconstruction_l3_alignment_gate")
set(RECONSTRUCTION_L3_UPSTREAM_COLMAP "" CACHE FILEPATH
    "latest pinned upstream COLMAP executable for reconstruction_l3_alignment_gate")
set(RECONSTRUCTION_L3_UPSTREAM_SOURCE "" CACHE PATH
    "Source checkout at the pinned upstream revision used to build the L3 executable")
set(RECONSTRUCTION_L3_UPSTREAM_REVISION "" CACHE STRING
    "Immutable revision embedded in the upstream L3 executable")
set(RECONSTRUCTION_L3_THRESHOLDS "" CACHE FILEPATH
    "Calibrated JSON thresholds for reconstruction_l3_alignment_gate")
set(RECONSTRUCTION_L3_OUTPUT "${CMAKE_BINARY_DIR}/Testing/reconstruction_l3" CACHE PATH
    "Fresh output directory for reconstruction_l3_alignment_gate")
set(RECONSTRUCTION_L3_SEED "20261009" CACHE STRING
    "Fixed mapper seed for reconstruction_l3_alignment_gate")
set(RECONSTRUCTION_L3_HARNESS "cli-anything-acloudviewer" CACHE FILEPATH
    "CLI harness executable for reconstruction_l3_alignment_gate")
option(ALL_SOURCE_TARGET "Whether to create a target for all source files (for Visual Studio / XCode development)" OFF)
option(DOWNLOAD_ENABLED "Whether to enable (automatic) download of resources (requires Curl/OpenSSL)" ON)
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
    if (NOT TARGET gtest_main)
        include(${CloudViewer_3RDPARTY_DIR}/googletest/googletest.cmake)
    endif ()
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

# Boost: the compiled Boost libraries (filesystem/program_options/graph/system)
# stay on the system / ext_boost version — the Boost 1.92 library upgrade was
# reverted because the fork's CGAL dependency requires boost/mpl/if_c.hpp and
# the legacy boost::filesystem API, both removed in 1.92, and the compiled
# libraries carry no numeric contract for the COLMAP subtree. Only the
# header-only boost::unordered subset is vendored (see src/CMakeLists.txt).
# Boost: unconditionally resolved through find_package on every platform —
# Linux uses the distro packages (apt libboost-all-dev in CI/docker), macOS
# and Windows use the conda boost (1.84). The former ext_boost source build
# (1.73, AliceVision tarball) existed solely for -DGLIBCXX_USE_CXX11_ABI=OFF
# builds, which no CI/docker/conda configuration uses (the option defaults to
# ON); with an ABI=0 build the caller must supply a matching-ABI boost via
# BOOST_ROOT. The compiled libraries carry no numeric contract for the
# COLMAP subtree; the only header-only special case is the vendored
# boost::unordered subset for patch_match.cc (see src/CMakeLists.txt).
set(COLMAP_BOOST_COMPONENTS graph filesystem program_options system)
if (TESTS_ENABLED)
    list(APPEND COLMAP_BOOST_COMPONENTS unit_test_framework)
endif ()
find_package(Boost REQUIRED COMPONENTS ${COLMAP_BOOST_COMPONENTS})
# The vendored Boost 1.92 header subset (see src/CMakeLists.txt) shadows the
# system Boost headers for every ColmapLib C++ TU. The historical Boost >=
# 1.74 floor came from a serialization/nvp double definition between the
# vendor closure and system Boost < 1.74 (Ubuntu 20.04 / Boost 1.71):
# boost/core/serialization.hpp defining boost::serialization::nvp clashed
# with the legacy serialization/nvp.hpp pulled by boost/graph ->
# multi_index, AND a forward-declaration broke Boost >= 1.84 whose own
# multi_index/serialization headers instantiate serialization::nvp from
# core/serialization.hpp (macOS conda 1.84). Fixed at the root by the
# guard handshake in the vendor core/serialization.hpp + noexcept-less
# core/nvp.hpp make_nvp (see the NOTE there), so every system Boost from
# 1.71 up compiles, links and runs the mixing probe
# (.qoder/tmp/boost_mix_probe.cc; measured PASS on 1.71, 1.74 and the
# 1.84 consumer pattern).
# NOTE: Ceres / Glog / Gflags / Metis are NOT resolved through find_package.
# They keep the fork-wide from-source build shared with the rest of the
# project (3rdparty_ceres with ext_lapack/suitesparse, ext_glog, ext_gflags;
# metis ships inside ext_suitesparse) and reach this subtree through the
# imported 3rdparty_* targets of find_dependencies.cmake on all three
# platforms. The find_package lines below are upstream COLMAP leftovers and
# must stay commented out.
# find_package(Glog REQUIRED)
# find_package(Ceres REQUIRED)
# find_package(Metis REQUIRED)

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
        # ABI=0 builds must point BOOST_ROOT at a matching-ABI boost install;
        # there is no bundled source build anymore (see the find_package note
        # above).
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

    # AppleClang 21 rejects the broken `operator bool()` bodies shipped in
    # CGAL <= 5.6.1's BGL iterator.h at template-definition time; shadow the
    # header with a patched copy (no-op on fixed CGAL installs).  See
    # cmake/CGALBglIteratorPatch.cmake.
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/CGALBglIteratorPatch.cmake)
    acv_enable_cgal_bgl_iterator_patch()
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

if (RECONSTRUCTION_HIP_ENABLED AND BUILD_CUDA_MODULE)
    message(FATAL_ERROR
        "RECONSTRUCTION_HIP_ENABLED and BUILD_CUDA_MODULE are mutually exclusive. "
        "Configure the HIP PatchMatch runner with BUILD_CUDA_MODULE=OFF.")
endif ()

set(HIP_ENABLED OFF)
if (RECONSTRUCTION_HIP_ENABLED)
    # HIP is a toolchain dependency, but its imported targets are normalized
    # through 3rdparty just like the other reconstruction dependencies.
    include(${CloudViewer_3RDPARTY_DIR}/rocm/rocm.cmake)
    enable_language(HIP)
    set(HIP_ENABLED ON)
    add_compile_definitions(COLMAP_HIP_ENABLED)
    message(STATUS "Enabling HIP PatchMatch support (archs: ${CMAKE_HIP_ARCHITECTURES})")
elseif (BUILD_CUDA_MODULE)
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_ARCHITECTURES}")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        message(
            FATAL_ERROR "You must set CMAKE_CUDA_ARCHITECTURES to e.g. 'native', 'all-major', '70', etc. "
            "More information at https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html")
    endif()

    set(CUDA_ENABLED ON)
    add_compile_definitions(CUDA_ENABLED)

    # Upstream parity: no -D_FORCE_INLINES. The Ubuntu-16.04-era workaround
    # changes glibc inline expansion inside nvcc's host pass, which perturbs
    # the optimizer's FMA/vectorization contracts in the host-side pose math
    # (ComputeRelativePose/ComposeProjectionMatrix) by ~1 ULP and cascades
    # into divergent depth maps versus the upstream binary.

    # Do not show warnings if the architectures are deprecated.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

    message(STATUS "Enabling CUDA support (version: ${CUDAToolkit_VERSION},"
            " archs: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(CUDA_ENABLED OFF)
    message(STATUS "Disabling CUDA support")
endif ()
message(STATUS "Colmap CUDA enabled: ${CUDA_ENABLED}")
message(STATUS "Colmap HIP enabled: ${HIP_ENABLED}")

if (GUI_ENABLED)
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

# Upstream parity (cmake/FindDependencies.cmake GPU_ENABLED): SiftGPU-based
# feature extraction/matching is available whenever OpenGL or CUDA is.
set(GPU_ENABLED OFF)
if (OPENGL_ENABLED OR CUDA_ENABLED)
    add_compile_definitions(GPU_ENABLED)
    set(GPU_ENABLED ON)
    message(STATUS "Enabling GPU support (OpenGL: ${OPENGL_ENABLED}, CUDA: ${CUDA_ENABLED})")
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

if (DOWNLOAD_ENABLED)
    add_compile_definitions(COLMAP_DOWNLOAD_ENABLED)
endif ()

# Export the engine feature macros that are currently active so consumers
# outside this directory scope (the python pybind extension TUs, which
# include the same engine headers) can mirror them and stay ABI-identical.
# add_compile_definitions above is directory-scoped and does NOT reach
# sibling directories.
set(ACV_COLMAP_ENGINE_MACROS "")
if (OPENMP_ENABLED AND OPENMP_FOUND)
    list(APPEND ACV_COLMAP_ENGINE_MACROS OPENMP_ENABLED)
endif ()
if (CUDA_ENABLED)
    list(APPEND ACV_COLMAP_ENGINE_MACROS CUDA_ENABLED)
endif ()
if (GPU_ENABLED)
    list(APPEND ACV_COLMAP_ENGINE_MACROS GPU_ENABLED)
endif ()
if (GUI_ENABLED)
    list(APPEND ACV_COLMAP_ENGINE_MACROS GUI_ENABLED)
endif ()
if (OPENGL_ENABLED)
    list(APPEND ACV_COLMAP_ENGINE_MACROS OPENGL_ENABLED)
endif ()
if (CGAL_FOUND AND CGAL_ENABLED)
    list(APPEND ACV_COLMAP_ENGINE_MACROS CGAL_ENABLED)
endif ()
if (DOWNLOAD_ENABLED)
    list(APPEND ACV_COLMAP_ENGINE_MACROS COLMAP_DOWNLOAD_ENABLED)
endif ()
list(APPEND ACV_COLMAP_ENGINE_MACROS MAX_STURM_RECURSION_DEPTH_LIMIT=300)
list(APPEND ACV_COLMAP_ENGINE_MACROS POSELIB_DEBUG=0)
if (RECONSTRUCTION_FETCH_POSELIB)
    list(APPEND ACV_COLMAP_ENGINE_MACROS RECONSTRUCTION_FETCH_POSELIB)
endif ()
if (HIP_ENABLED)
    list(APPEND ACV_COLMAP_ENGINE_MACROS COLMAP_HIP_ENABLED)
endif ()
# Unconditional vendored-library configuration macros that engine headers
# propagate to their consumers; the python binding TUs must agree on these
# too (e.g. kiss_fft_scalar changes struct layouts).
list(APPEND ACV_COLMAP_ENGINE_MACROS BOOST_ALL_DYN_LINK)
list(APPEND ACV_COLMAP_ENGINE_MACROS BOOST_TEST_DYN_LINK)
list(APPEND ACV_COLMAP_ENGINE_MACROS CLOUDVIEWER_ENABLE_DLL_EXPORTS)
list(APPEND ACV_COLMAP_ENGINE_MACROS STB_IMAGE_IMPLEMENTATION)
list(APPEND ACV_COLMAP_ENGINE_MACROS STB_IMAGE_WRITE_IMPLEMENTATION)
list(APPEND ACV_COLMAP_ENGINE_MACROS TINYGLTF_IMPLEMENTATION)
list(APPEND ACV_COLMAP_ENGINE_MACROS TINYOBJLOADER_IMPLEMENTATION)
list(APPEND ACV_COLMAP_ENGINE_MACROS kiss_fft_scalar=double)
set(ACV_COLMAP_ENGINE_MACROS "${ACV_COLMAP_ENGINE_MACROS}"
        CACHE INTERNAL "ACloudViewer COLMAP engine feature macros")

# DA3 depth estimation integration (AICore from core/DA3)
if(AICore_ENABLED AND TARGET AICore)
    message(STATUS "DA3 depth estimation integration enabled (AICore)")
endif()

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
        3rdparty_openimageio
        )
else()
    set(COLMAP_EXTERNAL_LIBRARIES
        ${CMAKE_DL_LIBS}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        3rdparty_openimageio
        # ${GLOG_LIBRARIES}
        # ${CERES_LIBRARIES}
        )
endif()
        
if (OPENMP_FOUND)
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${OpenMP_libomp_LIBRARY})
endif ()

if (GUI_ENABLED)
    # Use unified Qt:: prefix (CMakeExternalLibs.cmake creates Qt:: aliases for Qt5)
    list(APPEND COLMAP_EXTERNAL_LIBRARIES Qt::Core Qt::OpenGL Qt::Widgets)
    if(USE_QT6)
        # Qt6: QOpenGLWidget is in OpenGLWidgets module
        list(APPEND COLMAP_EXTERNAL_LIBRARIES Qt::OpenGLWidgets)
    endif()
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

# PBA has been removed from the build: upstream COLMAP dropped the PBA
# bundle-adjustment backend (c9729fa0) for CUDA 12+ compatibility, and the
# aligned backend surface is Ceres + Caspar (see optim/bundle_adjustment.h).

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

COLMAP_ADD_SOURCE_DIR(lib/SiftGPU LIB_SIFT_GPU_SRCS *.h *.cpp *.cu)
COLMAP_ADD_SOURCE_DIR(lib/SQLite LIB_SQLITE_SRCS *.h *.c)
COLMAP_ADD_SOURCE_DIR(lib/VLFeat LIB_VLFEAT_SRCS *.h *.c *.tc)

COLMAP_ADD_SOURCE_DIR(src/image IMAGE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/controllers CONTROLLERS_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/estimators ESTIMATORS_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/exe EXE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/feature FEATURE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/geometry GEOMETRY_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/mvs MVS_SRCS *.h *.cc *.cu)
COLMAP_ADD_SOURCE_DIR(src/optim OPTIM_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/math MATH_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/retrieval RETRIEVAL_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/scene SCENE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/sfm SFM_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/sensor SENSOR_SRCS *.h *.cc)
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
            ${LIB_SIFT_GPU_SRCS}
            ${LIB_SQLITE_SRCS}
            ${LIB_VLFEAT_SRCS}
            ${IMAGE_SRCS}
            ${CONTROLLERS_SRCS}
            ${ESTIMATORS_SRCS}
            ${EXE_SRCS}
            ${FEATURE_SRCS}
            ${GEOMETRY_SRCS}
            ${MATH_SRCS}
            ${MVS_SRCS}
            ${OPTIM_SRCS}
            ${RETRIEVAL_SRCS}
            ${SCENE_SRCS}
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
