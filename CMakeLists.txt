cmake_minimum_required(VERSION 3.19)
# If you're using Ubuntu 18.04, we suggest you install the latest CMake from the
# official repository https://apt.kitware.com/.  CMake 3.18+ is required to
# allow linking with OBJECT libraries, to prevent erroneous -gencode option
# deduplication with CUDA, and to simplify generator expressions for selecting
# compile flags and setting global hardened link flags.

# One shouldn't generate the BUILD project directly in the SOURCES folder!
if (${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    if (NOT SAME_BUILD_AND_SOURCE_FOLDER_WARNING_ALREADY_ISSUED)
        message(FATAL_ERROR "It is not advised to BUILD the binaries directly in the SOURCE folder!\n"
                "If you want to proceed with this option, just CONFIGURE the project once again")
        set(SAME_BUILD_AND_SOURCE_FOLDER_WARNING_ALREADY_ISSUED TRUE)
    endif ()
endif ()

if (APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING
            "Minimum OS X deployment version" FORCE)
endif ()

if (UNIX AND NOT APPLE)
    # CMAKE_HOST_SYSTEM_PROCESSOR is only available after calling project(),
    # which depends on ${CLOUDVIEWER_VERSION}, which depends on ${DEVELOPER_BUILD}.
    execute_process(COMMAND uname -p
            OUTPUT_VARIABLE PROCESSOR_ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            )
    if (PROCESSOR_ARCH STREQUAL "aarch64")
        set(LINUX_AARCH64 TRUE)
    endif ()

    set(CMD "lsb_release" "-a")
    execute_process(
        COMMAND ${CMD}
        OUTPUT_VARIABLE LSB_RELEASE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "LSB Release: ${LSB_RELEASE}")
    string(REGEX MATCH "Release:	[0-9.]*" UBUNTU_VERSION ${LSB_RELEASE})
    string(REPLACE "Release:" "" UBUNTU_VERSION "${UBUNTU_VERSION}")
    string(STRIP "${UBUNTU_VERSION}" UBUNTU_VERSION)
    message(STATUS "Ubuntu Version: ${UBUNTU_VERSION}")
endif ()

# CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/CMake")

# set build time
string(TIMESTAMP COMPILE_TIME %Y-%m-%d)
set(BUILD_TIME  ${COMPILE_TIME})

# set git commit id and git commit date
include(GenerateVersionDefinitions)
message(STATUS "GIT_COMMIT_ID: " ${GIT_COMMIT_ID})
message(STATUS "GIT_COMMIT_DATE : " ${GIT_COMMIT_DATE})

if(APPLE)
    execute_process(COMMAND uname
            OUTPUT_VARIABLE MACOS_NAME
            OUTPUT_STRIP_TRAILING_WHITESPACE
            )
    execute_process(COMMAND uname -m
            OUTPUT_VARIABLE PROCESSOR_ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            )
    if(PROCESSOR_ARCH STREQUAL "arm64")
        set(APPLE_AARCH64 TRUE)
    endif()
endif()

include(CMakeDependentOption)

# ACloudViewer build options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(BUILD_EXAMPLES "Build CloudViewer example programs" ON)
option(BUILD_UNIT_TESTS "Build CloudViewer unit tests" OFF)
option(BUILD_BENCHMARKS "Build the micro benchmarks" OFF)
option(BUILD_OPENCV "Build the opencv" OFF)
option(BUILD_PYTHON_MODULE "Build the python module" ON)
option(BUILD_CUDA_MODULE "Build the CUDA module" OFF)
option(BUILD_COMMON_CUDA_ARCHS "Build for common CUDA GPUs (for release)" OFF)
option(BUILD_CACHED_CUDA_MANAGER "Build the cached CUDA memory manager" ON)
option(BUILD_GUI "Builds CloudViewerApp GUI" ON)
option(WITH_OPENMP "Use OpenMP multi-threading" ON)
option(WITH_IPPICV "Use Intel Performance Primitives" ON)
option(ENABLE_HEADLESS_RENDERING "Use OSMesa for headless rendering" OFF)
option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" OFF)
option(GLIBCXX_USE_CXX11_ABI "Set -D_GLIBCXX_USE_CXX11_ABI=0" OFF)
if(BUILD_GUI AND (WIN32 OR UNIX AND NOT LINUX_AARCH64 AND NOT APPLE_AARCH64))
    option(BUILD_WEBRTC           "Build WebRTC visualizer"                  ON )
else()
    option(BUILD_WEBRTC           "Build WebRTC visualizer"                  OFF)
endif ()
option(BUILD_JUPYTER_EXTENSION "Build Jupyter, requires BUILD_WEBRTC=ON" OFF)

# 3rd-party build options
option(USE_SIMD "Use Single Instruction Multiple Data speed optimization" OFF)
option(USE_SYSTEM_OPENCV "Use system pre-installed opencv" OFF)
option(USE_SYSTEM_EIGEN3 "Use system pre-installed eigen3" OFF)
option(USE_SYSTEM_FLANN "Use system pre-installed flann" OFF)
option(USE_SYSTEM_NANOFLANN "Use system pre-installed nanoflann"       OFF)
option(USE_SYSTEM_FMT "Use system pre-installed fmt" OFF)
option(USE_SYSTEM_GLEW "Use system pre-installed glew" OFF)
option(USE_SYSTEM_GLFW "Use system pre-installed glfw" OFF)
option(USE_SYSTEM_GOOGLETEST "Use system pre-installed googletest" OFF)
option(USE_SYSTEM_IMGUI "Use system pre-installed imgui" OFF)
option(USE_SYSTEM_JPEG "Use system pre-installed jpeg" OFF)
option(USE_SYSTEM_LIBLZF "Use system pre-installed liblzf" OFF)
option(USE_SYSTEM_PNG "Use system pre-installed png" OFF)
option(USE_SYSTEM_PYBIND11 "Use system pre-installed pybind11" OFF)
option(USE_SYSTEM_QHULLCPP "Use system pre-installed qhullcpp" OFF)
option(USE_SYSTEM_TINYGLTF "Use system pre-installed tinygltf" OFF)
option(USE_SYSTEM_TINYOBJLOADER "Use system pre-installed tinyobjloader" OFF)
option(USE_SYSTEM_FFMPEG "Use system pre-installed ffmpeg" OFF)
# 3rd-party build options
if(LINUX_AARCH64 OR APPLE_AARCH64)
    option(USE_BLAS               "Use BLAS/LAPACK instead of MKL"           ON )
else()
    option(USE_BLAS               "Use BLAS/LAPACK instead of MKL"           OFF)
endif()
if(USE_BLAS)
    option(USE_SYSTEM_BLAS        "Use system pre-installed openblas"        OFF)
else()
    option(USE_SYSTEM_BLAS        "Use system pre-installed openblas"        ON )
endif()

if(LINUX_AARCH64 OR APPLE_AARCH64)
    option(BUILD_FILAMENT_FROM_SOURCE "Build filament from source"           ON )
else()
    option(BUILD_FILAMENT_FROM_SOURCE "Build filament from source"           OFF)
endif()
option(PREFER_OSX_HOMEBREW "Prefer Homebrew libs over frameworks" ON)

# Sensor options
option(BUILD_LIBREALSENSE "Build support for Intel RealSense camera" OFF)
option(USE_SYSTEM_LIBREALSENSE "Use system pre-installed librealsense" OFF)
option(BUILD_AZURE_KINECT "Build support for Azure Kinect sensor" OFF)

# ML library options
option(BUILD_TENSORFLOW_OPS "Build ops for Tensorflow" OFF)
option(BUILD_PYTORCH_OPS "Build ops for Pytorch" OFF)
option(BUNDLE_CLOUDVIEWER_ML "Includes the CloudViewer-ML repo in the wheel" OFF)

option(BUILD_RECONSTRUCTION "Build the 3D Reconstruction" OFF)
if (NOT ${BUILD_RECONSTRUCTION})
    set(BUILD_COLMAP_GUI OFF)
endif()

option(PACKAGE "Package the build results with zip or not" OFF)

# PCL ENGINE SUPPORT
option(USE_PCL_BACKEND "Check to use pcl library as backend" ON)
option(WITH_PCL_NURBS "Use Pcl on_nurbs module or not" OFF)

# other options
option(CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING "change all dynamic dll output path to main project output path for debugging!" ON)
# if Boost is compiled without C++11 support (as it is often the case in OS packages) this must be enabled to avoid symbol conflicts (SCOPED_ENUM)
option(BOOST_NO_CXX11 "compiled without C++11 support" OFF)

# Release build options
option(DEVELOPER_BUILD "Add +commit_hash to the project version number" ON)
if (NOT DEVELOPER_BUILD)
    if (NOT BUILD_COMMON_CUDA_ARCHS)
        set(BUILD_COMMON_CUDA_ARCHS ON CACHE BOOL "Build for common CUDA GPUs (for release)" FORCE)
        message(WARNING "Setting BUILD_COMMON_CUDA_ARCHS=ON since DEVELOPER_BUILD is OFF.")
    endif ()
endif ()

option(BUILD_WITH_CONDA "Build with conda env" OFF)
if (BUILD_WITH_CONDA)
    # Setup conda env if possible 
    SET(CONDA_PREFIX "CONDA_PREFIX_NOT_FOUND" CACHE STRING "The building conda env path prefix.")
    if(NOT "${CONDA_PREFIX}" STREQUAL "CONDA_PREFIX_NOT_FOUND" AND EXISTS "${CONDA_PREFIX}")
        message("USER CONDA_PREFIX: ${CONDA_PREFIX}")
    elseif(DEFINED ENV{CONDA_PREFIX})
        message("ENV CONDA_PREFIX: $ENV{CONDA_PREFIX}")
        set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")
    else()
        message(FATAL_ERROR "FAILED TO ENSURE CONDA_PREFIX VARIABLES")
    endif()
    message(STATUS "Building with conda env at ${CONDA_PREFIX}")
endif()

# Default build type on single-config generators.
# For multi-config generators (e.g. Visual Studio), CMAKE_CONFIGURATION_TYPES
# will be set, and we don't specify a default CMAKE_BUILD_TYPE.
# https://blog.kitware.com/cmake-and-the-default-build-type/
if (NOT CMAKE_CONFIGURATION_TYPES)
    if (NOT CMAKE_BUILD_TYPE)
        message(STATUS "Setting build type to Release as none was specified.")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
        # Set the possible values of build type for cmake-gui.
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
    endif ()
    message(STATUS "CMAKE_BUILD_TYPE is set to ${CMAKE_BUILD_TYPE}.")
endif ()

find_program(CCACHE "ccache")
if (CCACHE)
    message(STATUS "ccache found at ${CCACHE}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    if (BUILD_CUDA_MODULE)
        set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE})
    endif ()
endif ()

# In ExternalProject_Add, if CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR is specified, CMake will
# use this directory to cache downloaded 3rd party dependencies and automatically skip
# downloading from the Internet if the files are available. This is only supported by
# a limited number of 3rd party libraries.
set(CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty_downloads")
message(STATUS "Downloading 3rdparty dependencies to ${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}")

set(FILAMENT_PRECOMPILED_ROOT "" CACHE PATH "Path to precompiled Filament library (used if BUILD_FILAMENT_FROM_SOURCE=OFF)")

if (PREFER_OSX_HOMEBREW)
    set(CMAKE_FIND_FRAMEWORK LAST)
    set(CMAKE_FIND_APPBUNDLE LAST)
endif ()

# Set OpenGL policy
if (NOT USE_SYSTEM_GLFW)
    cmake_policy(SET CMP0072 OLD)
endif ()
cmake_policy(GET CMP0072 CMP0072_VALUE)

# Set BLAS/LAPACK library preference.
if (LINUX_AARCH64)
    if (NOT USE_BLAS)
        set(USE_BLAS ON)
        message(WARNING "ARM CPU detected, setting USE_BLAS=ON.")
    endif ()
    if (BUILD_GUI AND (NOT BUILD_FILAMENT_FROM_SOURCE))
        set(BUILD_FILAMENT_FROM_SOURCE ON)
        message(WARNING "ARM CPU detected, setting BUILD_FILAMENT_FROM_SOURCE=ON.")
    endif ()
endif ()

# Catch a few incompatible build options
if ((LINUX_AARCH64 OR APPLE_AARCH64) AND NOT BUILD_FILAMENT_FROM_SOURCE)
    message(FATAL_ERROR "ARM CPU detected, you must set BUILD_FILAMENT_FROM_SOURCE=ON.")
endif()
if ((LINUX_AARCH64 OR APPLE_AARCH64) AND NOT USE_BLAS)
    message(FATAL_ERROR "ARM CPU detected, you must set USE_BLAS=ON.")
endif()
if (APPLE AND ENABLE_HEADLESS_RENDERING)
    message(WARNING "Headless rendering is not supported on Mac OS")
    set(ENABLE_HEADLESS_RENDERING OFF)
endif ()
if (ENABLE_HEADLESS_RENDERING AND BUILD_GUI)
    message(WARNING "Headless rendering disables the CloudViewer GUI")
    set(BUILD_GUI OFF)
endif ()
if (ENABLE_HEADLESS_RENDERING AND (USE_SYSTEM_GLEW OR USE_SYSTEM_GLFW))
    message(WARNING "Headless rendering requires customized GLEW and GLFW builds")
    set(USE_SYSTEM_GLEW OFF)
    set(USE_SYSTEM_GLFW OFF)
endif ()
if (BUNDLE_CLOUDVIEWER_ML AND NOT (BUILD_TENSORFLOW_OPS OR BUILD_PYTORCH_OPS))
    message(SEND_ERROR "3DML depends on TensorFlow or PyTorch Ops. Enable them with -DBUILD_TENSORFLOW_OPS=ON or -DBUILD_PYTORCH_OPS=ON")
endif ()
if (BUILD_WEBRTC AND LINUX_AARCH64)
    message(FATAL_ERROR "BUILD_WEBRTC=ON is not yet supported on ARM Linux")
endif ()
if (BUILD_WEBRTC AND NOT BUILD_GUI)
    message(FATAL_ERROR "BUILD_WEBRTC=ON requires BUILD_GUI=ON")
endif ()
if (BUILD_JUPYTER_EXTENSION AND NOT BUILD_WEBRTC)
    # BUILD_JUPYTER_EXTENSION transitively depends on BUILD_GUI
    message(FATAL_ERROR "BUILD_JUPYTER_EXTENSION=ON requires BUILD_WEBRTC=ON")
endif ()
if (BUILD_JUPYTER_EXTENSION AND NOT BUILD_PYTHON_MODULE)
    message(FATAL_ERROR "BUILD_JUPYTER_EXTENSION=ON requires BUILD_PYTHON_MODULE=ON")
endif ()

# Parse CloudViewer version number
file(STRINGS "libs/cloudViewer/version.txt" CLOUDVIEWER_VERSION_READ)
foreach (ver ${CLOUDVIEWER_VERSION_READ})
    if (ver MATCHES "CLOUDVIEWER_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(CLOUDVIEWER_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif ()
endforeach ()
set(CLOUDVIEWER_VERSION_DEVHASH "")
if (DEVELOPER_BUILD)
    execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" log --pretty=format:%h -n 1
            OUTPUT_VARIABLE GIT_REV)
    if (GIT_REV)
        set(CLOUDVIEWER_VERSION_DEVHASH "+${GIT_REV}")
    endif ()
endif ()
string(CONCAT CLOUDVIEWER_VERSION
        "${CLOUDVIEWER_VERSION_MAJOR}"
        ".${CLOUDVIEWER_VERSION_MINOR}"
        ".${CLOUDVIEWER_VERSION_PATCH}"
        )

# npm version has to be MAJOR.MINOR.PATCH
string(CONCAT PROJECT_VERSION_THREE_NUMBER "${CLOUDVIEWER_VERSION_MAJOR}"
        ".${CLOUDVIEWER_VERSION_MINOR}"
        ".${CLOUDVIEWER_VERSION_PATCH}")
set(CLOUDVIEWER_VERSION_FULL "${CLOUDVIEWER_VERSION}${CLOUDVIEWER_VERSION_DEVHASH}" CACHE
    STRING "ACloudViewer full version.")
set(CLOUDVIEWER_ABI_VERSION "${PROJECT_VERSION_THREE_NUMBER}" CACHE
    STRING "ACloudViewer ABI version / SOVERSION (for releases only).")

project(ACloudViewer 
    VERSION ${CLOUDVIEWER_VERSION}
    # Set PROJECT_DESCRIPTION
    DESCRIPTION "ACloudViewer: A Modern Library for 3D Data Processing."
    # Set PROJECT_HOMEPAGE_URL
    HOMEPAGE_URL "https://asher-1.github.io"
    LANGUAGES C CXX)
message(STATUS "ACloudViewer ${CLOUDVIEWER_VERSION_FULL}")

# Setup Python executable
# if(NOT DEFINED Python3_FIND_REGISTRY)
#     # Only consider PATH variable on Windows by default
#     set(Python3_FIND_REGISTRY NEVER)
# endif()
# Requires Python 3.6+
find_package(Python3 3.6
        COMPONENTS Interpreter Development
        )
if (Python3_FOUND)
    # Setup PYTHON_EXECUTABLE for 3rdparty modules
    # which still use the deprecated find_package(PythonInterp)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE} CACHE FILEPATH
            "Deprecated path to the Python executable (for 3rdparty only)")

    # fix always pybind.cpython-36m-x86_64-linux-gnu.so regardless given other python version(eg: python2.7, python3.8)
    include(FindPythonLibsNew)
    set(PYTHON_INCLUDE_DIR ${PYTHON_INCLUDE_DIRS} CACHE PATH
            "Deprecated path to the Python include dir (for pybind only)")
else ()
    if (BUILD_PYTHON_MODULE)
        message(FATAL_ERROR "BUILD_PYTHON_MODULE=ON requires Python >= 3.6. Please ensure it is in PATH.")
    endif ()
endif ()

# PyPI package name controls specifies the repository name on PyPI. The default
# name is "cloudViewer". In the past, for historical reasons, we've used the
# following names for PyPI, while they are now deprecated:
# - cloudViewer-python
# - py3d
# - cloudViewer-original
# - cloudViewer-official
# - cloud-viewer
if (NOT DEFINED PYPI_PACKAGE_NAME)
    set(PYPI_PACKAGE_NAME "cloudViewer")
endif ()

# set additional info
set(PROJECT_EMAIL "dahailu@foxmail.com")
set(PROJECT_HOME "https://asher-1.github.io")
set(PROJECT_DOCS "https://asher-1.github.io/docs")
set(PROJECT_CODE "https://github.com/Asher-1/ACloudViewer")
set(PROJECT_ISSUES "https://github.com/Asher-1/ACloudViewer/issues")

include(CMakePolicies)
include(CMakeSetCompilerOptions)
include(DeployQt)
# Load advanced scripts
include(CMakeInclude)
include(Install)

# Set installation paths
if (UNIX OR CYGWIN)
    include(GNUInstallDirs)
    set(CloudViewer_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
    set(CloudViewer_INSTALL_BIN_DIR "${CMAKE_INSTALL_BINDIR}")
    set(CloudViewer_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}")
    # Put resources in */share/
    set(CloudViewer_INSTALL_RESOURCE_DIR "${CMAKE_INSTALL_DATADIR}")
    set(CloudViewer_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
else ()
    set(CloudViewer_INSTALL_INCLUDE_DIR include)
    set(CloudViewer_INSTALL_BIN_DIR bin)
    set(CloudViewer_INSTALL_LIB_DIR lib)
    # Put resources in */bin, with executables / DLLs
    set(CloudViewer_INSTALL_RESOURCE_DIR bin)
    set(CloudViewer_INSTALL_CMAKE_DIR "share/CloudViewer")
endif ()

# RPATH Linux/Unix: (dynamic) lib are put in a subdir of prefix/lib,
# since they are only used by libs/CVViewer
# if (UNIX AND NOT APPLE)
#     include(GNUInstallDirs)
#     set(CMAKE_INSTALL_RPATH "lib")
# endif ()

# RPATH handling (for TBB DSO). Check current folder, one folder above and the lib sibling folder 
# set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
# if (APPLE)
#     # Add options to cover the various ways in which ACloudViewer shared lib or apps can be installed wrt TBB DSO
#     set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../;@loader_path/../lib/")
#     # pybind with ACloudViewer shared lib is copied, not cmake-installed, so we need to add .. to build rpath 
#     set(CMAKE_BUILD_RPATH "@loader_path/../")
# elseif(UNIX)
#     set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../;$ORIGIN/../lib/")
#     set(CMAKE_BUILD_RPATH "$ORIGIN/../")
# endif()

# Define target folders
# (now that ccViewer can have its own plugins, ACloudViewer and CloudViewer must fall in separate folders!
set(ACloudViewer_DEST_FOLDER "${CloudViewer_INSTALL_BIN_DIR}/ACloudViewer")

# set default install folders
if (WIN32)
    # (on Windows - msvc generator - the '_debug' suffix is automatically added for debug configurations)
    set(INSTALL_DESTINATIONS ${CMAKE_INSTALL_PREFIX}/${ACloudViewer_DEST_FOLDER})
elseif (UNIX OR APPLE)
    set(INSTALL_DESTINATIONS ${CMAKE_INSTALL_PREFIX})
endif ()

# Put build results in some predictable places
# The $<CONFIG> generator expression makes sure that XCode or Visual Studio do not
# append additional path components, as we need to know *exactly* where the build results
# end up.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(COLOUDVIEWER_PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(COLOUDVIEWER_APP_DIR_NAME "CloudViewer")
set(CLOUDVIEWER_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${COLOUDVIEWER_APP_DIR_NAME}")
if (NOT APPLE)
    file(MAKE_DIRECTORY ${CLOUDVIEWER_OUTPUT_DIRECTORY})
endif ()

# Global Security options (including 3rd party code)
# Add -fPIC for library and -fPIE for executable to compiler and linker. Does not add -pie !
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Explicitly specify the preference of using -pthread over -lpthread.
# This must be defined here since CUDA calls find_package(Threads) internally.
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# Overwrites property for Thread::Threads in find_package(Threads)
# For CUDA, "-pthread" is replaced with "-Xcompiler -pthread" (CMake's default)
# For ISPC, "-pthread" is disabled
macro(cloudViewer_patch_findthreads_module_)
    if(TARGET Threads::Threads AND THREADS_HAVE_PTHREAD_ARG)
        set_property(TARGET Threads::Threads
                PROPERTY INTERFACE_COMPILE_OPTIONS
                "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:SHELL:-Xcompiler -pthread>"
                "$<$<AND:$<NOT:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>>,$<NOT:$<COMPILE_LANGUAGE:ISPC>>>:-pthread>")
    endif()
endmacro()
cmake_language(EVAL CODE "cmake_language(DEFER CALL cloudViewer_patch_findthreads_module_)")

# Build CUDA module by default if CUDA is available
if (BUILD_CUDA_MODULE)
    # Suppress warnings for deprecated C++17 functions (stdgpu->thrust with CUDA 11 for MSVC).
    # add_compile_definitions($<$<COMPILE_LANGUAGE:CUDA>:_ALLOW_UNSUPPORTED_LIBCPP>)
    add_compile_definitions($<$<COMPILE_LANGUAGE:CUDA>:_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING>)
    include(CloudViewerMakeCudaArchitectures)
    cloudViewer_make_cuda_architectures(CUDA_ARCHS)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS})
    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "10.1")
        message(FATAL_ERROR "CUDA 10.0 and older are not supported. Please upgrade to CUDA 10.1 or newer.")
    endif ()
endif ()

# OS specific settings
if (WIN32)
    # Windows defaults to hidden symbol visibility, override that
    # TODO: It would be better to explicitly export symbols.
    # Then, we could use -fvisibility=hidden for Linux as well
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if (MSVC)
        # Make sure we don't hit the 65535 object member limit with MSVC
        #
        # /bigobj allows object files with more than 65535 members
        # /Ob2 enables function inlining, because MSVC is particularly
        # verbose with inline members
        #
        # See: https://github.com/tensorflow/tensorflow/pull/10962
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj;/Ob2>")
    endif ()
    #	if (STATIC_WINDOWS_RUNTIME)
    #        # This needs cmake_policy(SET CMP0091 NEW)
    #        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded<<CONFIG:Debug>:Debug>")
    #    endif()
    if (STATIC_WINDOWS_RUNTIME)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else ()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif ()
endif ()

# Downloads files from https://github.com/Asher-1/cloudViewer_downloads to
# CloudViewer/examples/test_data.
include(${PROJECT_SOURCE_DIR}/examples/cloudViewer_downloads.cmake)

# Folder view for project files
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Check if the compiler defines the _GLIBCXX_USE_CXX11_ABI macro
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("#include <cxxabi.h>
int main() { return _GLIBCXX_USE_CXX11_ABI; }" HAS_GLIBCXX_USE_CXX11_ABI)

macro(add_source_group module_name)
    file(GLOB MODULE_HEADER_FILES "${module_name}/*.h")
    source_group("Header Files\\${module_name}" FILES ${MODULE_HEADER_FILES})
    file(GLOB MODULE_SOURCE_FILES "${module_name}/*.cpp")
    source_group("Source Files\\${module_name}" FILES ${MODULE_SOURCE_FILES})
    file(GLOB MODULE_SHADER_FILES "${module_name}/*.glsl")
    source_group("Source Files\\Shader\\GLSL" FILES ${MODULE_SHADER_FILES})
    file(GLOB MODULE_MATERIAL_FILES "${module_name}/*.mat")
    source_group("Source Files\\Material" FILES ${MODULE_MATERIAL_FILES})
endmacro()

if (BUILD_RECONSTRUCTION)
    message(STATUS "Enabling Reconstruction support and automatically enable SIMD!")
    set(USE_SIMD ON)
endif ()

# check for SSE flags
include(CMakeFindSSE)
if (USE_SIMD)
    CLOUDVIEWER_CHECK_FOR_SSE()
    set(CLOUDVIEWERCONFIG_SSE_DEFINITIONS "${SSE_DEFINITIONS}")
    set(CLOUDVIEWERCONFIG_SSE_COMPILE_OPTIONS ${SSE_FLAGS})
    if (MSVC)
        add_definitions(${CLOUDVIEWERCONFIG_SSE_DEFINITIONS})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS_STR}")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS_STR}")
    endif ()

    message(STATUS "Enabling SIMD support")
    message(STATUS "Using CPU native flags for SSE optimization: ${CLOUDVIEWERCONFIG_SSE_DEFINITIONS}")
else ()
    message(STATUS "Disabling SIMD support")
endif ()

# Include convenience functions
include(CloudViewerLink3rdpartyLibraries)
include(CloudViewerSetGlobalProperties)
include(CloudViewerShowAndAbortOnWarning)

# Add external libraries
include(CMakeExternalLibs)

# Enumerate all third-party libraries which we need later
# This creates the necessary targets and sets the
# CloudViewer_3RDPARTY_*_TARGETS variables we use in cloudViewer_link_3rdparty_libraries
include(3rdparty/find_dependencies.cmake)

# Default debug suffix for libraries.
# Note: must put after include(3rdparty/find_dependencies.cmake)
set(CMAKE_DEBUG_POSTFIX "d")

# Contrib. libraries (mainly for I/O)
include(AllSupport)

SET(PROJECT_ROOT_PATH "${PROJECT_SOURCE_DIR}")

if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
    if (MSVC)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin/RelWithDebInfo)
    else ()
        SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endif ()
endif ()

add_subdirectory(core)

# Propagate the choice to use doubles for ScalarType
if (${CV_CORE_LIB_USES_DOUBLE})
    add_definitions(-DSCALAR_TYPE_DOUBLE)
else ()
    add_definitions(-DSCALAR_TYPE_FLOAT)
endif ()

# Internal libs used by both ACloudViewer and cloudViewer python package
add_subdirectory(libs)

# Examples
if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
    if (MSVC)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/examples/Debug)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/examples/Release)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin/examples/RelWithDebInfo)
    else ()
        SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endif ()
endif ()
add_subdirectory(examples)

# load plugins (potentially used by eCV)
if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
    if (MSVC)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug/plugins)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release/plugins)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin/RelWithDebInfo/plugins)
    else ()
        SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins)
    endif ()
endif ()
add_subdirectory(plugins)

# eCV
if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
    if (MSVC)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin/RelWithDebInfo)
    else ()
        SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endif ()
endif ()

# Set apps name
set(MAIN_APP_NAME "ACloudViewer")
if (BUILD_GUI)
    set(CLOUDVIEWER_APP_NAME ${COLOUDVIEWER_APP_DIR_NAME})
endif()
if (BUILD_RECONSTRUCTION)
    if (APPLE)
        set(COLMAP_APP_NAME "colmap")
    elseif (UNIX)
        set(COLMAP_APP_NAME "colmap_app")
    elseif (WIN32)
        set(COLMAP_APP_NAME "colmap_app")
    endif()
endif()

# ACloudViewer app
add_subdirectory(eCV)

# Install CMake configuration files
install(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}:: DESTINATION ${CloudViewer_INSTALL_CMAKE_DIR})
export(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}::)

if (Python3_EXECUTABLE)
    # `make check-style` checks style for c++/cuda/python/ipynb files
    add_custom_target(check-style
            COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_style.py
            COMMENT "Python executable used for style check: ${Python3_EXECUTABLE}."
            )

    # `make apply-style` applies style for c++/cuda/python/ipynb files
    add_custom_target(apply-style
            COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_style.py --do_apply_style
            COMMENT "Python executable used for style check: ${Python3_EXECUTABLE}."
            )
endif ()

# `make check-cpp-style` checks style for c++/cuda files.
# This works outside of python virtualenv.
add_custom_target(check-cpp-style
        COMMAND ${CMAKE_COMMAND}
        -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        -DDO_APPLY_STYLE=OFF
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_cpp_style.cmake
        )

# `make apply-cpp-style` applies style for c++/cuda files.
# This works outside of python virtualenv.
add_custom_target(apply-cpp-style
        COMMAND ${CMAKE_COMMAND}
        -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        -DDO_APPLY_STYLE=ON
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_cpp_style.cmake
        )

include(CloudViewerPrintConfigurationSummary)
cloudViewer_print_configuration_summary()

### Install files and library
set(SCRIPTS_BASE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
file(GLOB main_documents_list "*.md" )
list(APPEND main_documents_list "CHANGELOG.txt" )
list(APPEND main_documents_list "doc/cloudviewer-detail-zh.md" )
list(APPEND main_documents_list "eCV/bin_other/license.txt" 
                                "eCV/bin_other/global_shift_list_template.txt")

SET(DEPLOY_ROOT_PATH "${INSTALL_DESTINATIONS}/deploy")
install(DIRECTORY "${SCRIPTS_BASE_PATH}/QtInstallerFramework/deploy"
        DESTINATION "${INSTALL_DESTINATIONS}"
        USE_SOURCE_PERMISSIONS)

if (BUILD_CUDA_MODULE)
    set(ACLOUDVIEWER_DEVICE "cuda")
else ()
    set(ACLOUDVIEWER_DEVICE "cpu")
endif()

set(colmap_helper_utils "")
set(cloudviewer_helper_utils "")

if (APPLE)
    if (APPLE_AARCH64)
        set(ARC_NAME "ARM64")
    else()
        set(ARC_NAME "X64")
    endif()
    set(LIBS_FOLDER_NAME "Frameworks")
    set(PACK_SCRIPTS_PATH "${SCRIPTS_BASE_PATH}/platforms/mac")
    set(CLOUDVIEWER_PACKAGE_NAME "${MAIN_APP_NAME}-${CLOUDVIEWER_VERSION_FULL}-${BUILD_TIME}-${ARC_NAME}")
elseif (UNIX)
    if (LINUX_AARCH64)
        set(ARC_NAME "arm64")
    else()
        set(ARC_NAME "amd64")
    endif()
    set(LIBS_FOLDER_NAME "${CloudViewer_INSTALL_LIB_DIR}")
    set(PACK_SCRIPTS_PATH "${SCRIPTS_BASE_PATH}/platforms/linux")
    set(CLOUDVIEWER_PACKAGE_NAME "${MAIN_APP_NAME}-${CLOUDVIEWER_VERSION_FULL}-${BUILD_TIME}-ubuntu${UBUNTU_VERSION}-${ACLOUDVIEWER_DEVICE}-${ARC_NAME}")

    # for ACloudViewer
    file(GLOB main_helper_utils "${SCRIPTS_BASE_PATH}/${MAIN_APP_NAME}/${MAIN_APP_NAME}*" )
    list(APPEND main_documents_list ${main_helper_utils})

    # for CloudViewer
    if (BUILD_GUI)
        file(GLOB cloudviewer_helper_files "${SCRIPTS_BASE_PATH}/${CLOUDVIEWER_APP_NAME}/${CLOUDVIEWER_APP_NAME}*" )
        list(APPEND cloudviewer_helper_utils ${cloudviewer_helper_files})
    endif()

    # for colmap
    if (BUILD_RECONSTRUCTION)
        file(GLOB colmap_helper_files "${SCRIPTS_BASE_PATH}/${COLMAP_APP_NAME}/${COLMAP_APP_NAME}*" )
        list(APPEND colmap_helper_utils ${colmap_helper_files} "${SCRIPTS_BASE_PATH}/${COLMAP_APP_NAME}/Colmap.desktop")
        # fix gflags undefinded symbols issues
        set(src_gflags_filename ${CMAKE_SHARED_LIBRARY_PREFIX}${EXT_GFLAGS_LIBRARIES}${CMAKE_SHARED_LIBRARY_SUFFIX}.${GFLAGS_MAJOR_VERSION}.${GFLAGS_MINOR_VERSION})
        set(target_gflags_filename ${CMAKE_SHARED_LIBRARY_PREFIX}${EXT_GFLAGS_LIBRARIES}${CMAKE_SHARED_LIBRARY_SUFFIX}.${GFLAGS_MINOR_VERSION})
        install(CODE "set(GFLAGS_SRC_FILENAME \"${src_gflags_filename}\")")
        install(CODE "set(GFLAGS_DST_FILENAME \"${target_gflags_filename}\")")
    endif()

    # for qt plugins
    foreach( qt5_plugins_folder ${QT5_PLUGINS_PATH_LIST} )
        install(DIRECTORY "${qt5_plugins_folder}"
                DESTINATION "${DEPLOY_ROOT_PATH}/packages/${MAIN_APP_NAME}/data" 
                USE_SOURCE_PERMISSIONS)
    endforeach()
elseif (WIN32)
    set(ARC_NAME "amd64")
    set(LIBS_FOLDER_NAME "${CloudViewer_INSTALL_LIB_DIR}")
    set(PACK_SCRIPTS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/scripts/platforms/windows")
    message(STATUS "Unsupported on windows now!")
endif()

# for ACloudViewer
foreach( filename ${main_documents_list} )
    cloudViewer_install_ext( FILES ${filename} "${DEPLOY_ROOT_PATH}/packages/${MAIN_APP_NAME}/data" "" )
endforeach()

# for CloudViewer
if (BUILD_GUI)
    foreach( filename ${cloudviewer_helper_utils} )
        cloudViewer_install_ext( FILES ${filename} "${DEPLOY_ROOT_PATH}/packages/${CLOUDVIEWER_APP_NAME}/data" "" )
    endforeach()
endif()

if (BUILD_RECONSTRUCTION)
    # for Colmap
    foreach( filename ${colmap_helper_utils} )
        cloudViewer_install_ext( FILES ${filename} "${DEPLOY_ROOT_PATH}/packages/colmap/data" "" )
    endforeach()
endif()

message(STATUS "CLOUDVIEWER_PACKAGE_NAME: " ${CLOUDVIEWER_PACKAGE_NAME})
# for linux
install(CODE "set(BUILD_LIB_PATH \"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}\")")
install(CODE "set(EXTERNAL_INSTALL_DIR \"${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}\")")
# for macos
install(CODE "set(CONDA_PREFIX \"${CONDA_PREFIX}\")")
install(CODE "set(BUILD_WITH_CONDA \"${BUILD_WITH_CONDA}\")")
# common
install(CODE "set(BUILD_GUI \"${BUILD_GUI}\")")
install(CODE "set(BUILD_RECONSTRUCTION \"${BUILD_RECONSTRUCTION}\")")
install(CODE "set(LIBS_FOLDER_NAME \"${LIBS_FOLDER_NAME}\")")
install(CODE "set(BUILD_TIME \"${BUILD_TIME}\")")
install(CODE "set(CLOUDVIEWER_VERSION \"${CLOUDVIEWER_VERSION_FULL}\")")
install(CODE "set(MAIN_APP_NAME \"${MAIN_APP_NAME}\")")
install(CODE "set(COLMAP_APP_NAME \"${COLMAP_APP_NAME}\")")
install(CODE "set(CLOUDVIEWER_APP_NAME \"${CLOUDVIEWER_APP_NAME}\")")
install(CODE "set(CloudViewer_INSTALL_BIN_DIR \"${CloudViewer_INSTALL_BIN_DIR}\")")
install(CODE "set(PACK_SCRIPTS_PATH \"${PACK_SCRIPTS_PATH}\")")
install(CODE "set(DEPLOY_ROOT_PATH \"${DEPLOY_ROOT_PATH}\")")
install(CODE "set(CLOUDVIEWER_PACKAGE_NAME \"${CLOUDVIEWER_PACKAGE_NAME}\")")
install(CODE "set(PACKAGE \"${PACKAGE}\")")
# call cmake script to deploy
install(SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PostInstall.cmake")
install(CODE "MESSAGE(\"${MAIN_APP_NAME}\.${CLOUDVIEWER_VERSION} install successfully.\")")