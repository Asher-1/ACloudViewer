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
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING
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
endif ()

include(CMakeDependentOption)

# ErowCloudViewer build options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(BUILD_EXAMPLES "Build CloudViewer example programs" ON)
option(BUILD_UNIT_TESTS "Build CloudViewer unit tests" OFF)
option(BUILD_BENCHMARKS "Build the micro benchmarks" OFF)
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
if (WIN32 OR UNIX AND NOT LINUX_AARCH64)
    cmake_dependent_option(BUILD_WEBRTC "Build WebRTC visualizer" ON "BUILD_GUI" OFF)
else ()
    option(BUILD_WEBRTC "Build WebRTC visualizer" OFF)
endif ()
option(BUILD_JUPYTER_EXTENSION "Build Jupyter, requires BUILD_WEBRTC=ON" OFF)

# 3rd-party build options
option(USE_SIMD "Use Single Instruction Multiple Data speed optimization" OFF)
option(USE_BLAS "Use BLAS/LAPACK instead of MKL" OFF)
option(USE_SYSTEM_EIGEN3 "Use system pre-installed eigen3" OFF)
option(USE_SYSTEM_FLANN "Use system pre-installed flann" OFF)
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
option(BUILD_FILAMENT_FROM_SOURCE "Build filament from source" OFF)
option(PREFER_OSX_HOMEBREW "Prefer Homebrew libs over frameworks" ON)
option(WITH_FAISS "Enable Faiss" ON)

# Sensor options
option(BUILD_LIBREALSENSE "Build support for Intel RealSense camera" OFF)
option(USE_SYSTEM_LIBREALSENSE "Use system pre-installed librealsense" OFF)
option(BUILD_AZURE_KINECT "Build support for Azure Kinect sensor" OFF)

# ML library options
option(BUILD_TENSORFLOW_OPS "Build ops for Tensorflow" OFF)
option(BUILD_PYTORCH_OPS "Build ops for Pytorch" OFF)
option(BUNDLE_CLOUDVIEWER_ML "Includes the CloudViewer-ML repo in the wheel" OFF)

# PCL ENGINE SUPPORT
option(BUILD_RECONSTRUCTION "Build the 3D Reconstruction" OFF)
option(USE_PCL_BACKEND "Check to use pcl library as backend" ON)

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
set(CLOUDVIEWER_VERSION_FULL "${CLOUDVIEWER_VERSION}${CLOUDVIEWER_VERSION_DEVHASH}")

project(ErowCloudViewerProjects VERSION ${CLOUDVIEWER_VERSION} LANGUAGES C CXX)
message(STATUS "ErowCloudViewerProjects ${CLOUDVIEWER_VERSION_FULL}")

# FIXME: Remove this workaround once a fixed Visual Studio 16.10 version is released.
if (BUILD_CUDA_MODULE
        AND CMAKE_CXX_COMPILER MATCHES "MSVC"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19.29"
        )
    # Keep C++14 standard for unaffected C++ files, but use C++17 for CUDA files.
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CUDA_STANDARD 17)
    # Suppress warnings for deprecated C++17 functions.
    add_compile_definitions($<$<COMPILE_LANGUAGE:CUDA>:_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING>)
    message(WARNING "Visual Studio 16.10 (MSVC 19.29) introduced a compiler bug when compiling CUDA code with C++14. "
            "Workaround this bug by setting the CUDA standard to C++17.")
endif ()

# CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/CMake")
set(ECV_CMAKE_SCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Setup Python executable
#if (NOT DEFINED Python3_FIND_REGISTRY)
#    # Only consider PATH variable on Windows by default
#    set(Python3_FIND_REGISTRY NEVER)
#endif ()
find_package(Python3 3.6...<3.10
        COMPONENTS Interpreter Development
        )
if (Python3_FOUND)
    # Setup Python3_EXECUTABLE for 3rdparty modules
    # which still use the deprecated find_package(PythonInterp)
    set(Python3_EXECUTABLE ${Python3_EXECUTABLE} CACHE FILEPATH
            "Deprecated path to the Python executable (for 3rdparty only)")

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
        message(FATAL_ERROR "BUILD_PYTHON_MODULE=ON requires Python 3.6-3.9. Please ensure it is in PATH.")
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
set(PROJECT_CODE "https://github.com/Asher-1/ErowCloudViewer")
set(PROJECT_ISSUES "https://github.com/Asher-1/ErowCloudViewer/issues")

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
    set(CloudViewer_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
else ()
    set(CloudViewer_INSTALL_INCLUDE_DIR include)
    set(CloudViewer_INSTALL_BIN_DIR bin)
    set(CloudViewer_INSTALL_LIB_DIR lib)
    set(CloudViewer_INSTALL_CMAKE_DIR "share/CloudViewer")
endif ()

# RPATH Linux/Unix: (dynamic) libs are put in a subdir of prefix/lib,
# since they are only used by libs/CVViewer
if (UNIX AND NOT APPLE)
    include(GNUInstallDirs)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}/erowcloudviewer")
endif ()

# Define target folders
# (now that ccViewer can have its own plugins, ErowCloudViewer and CloudViewer must fall in separate folders!
if (WIN32 OR APPLE)
    set(EROWCLOUDVIEWER_DEST_FOLDER "bin/ErowCloudViewer")
else ()
    set(EROWCLOUDVIEWER_DEST_FOLDER bin)
endif ()

# set default install folders
if (WIN32)
    # (on Windows - msvc generator - the '_debug' suffix is automatically added for debug configurations)
    set(INSTALL_DESTINATIONS ${EROWCLOUDVIEWER_DEST_FOLDER})
elseif (UNIX AND NOT APPLE)
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
make_directory(${CLOUDVIEWER_OUTPUT_DIRECTORY})

# Global Security options (including 3rd party code)
# Add -fPIC for library and -fPIE for executable to compiler and linker. Does not add -pie !
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CloudViewerMakeHardeningFlags)
cloudViewer_make_hardening_flags(HARDENING_CFLAGS HARDENING_LDFLAGS)
cloudViewer_make_hardening_definitions(HARDENING_DEFINITIONS)

message(STATUS "Using security hardening compiler flags: ${HARDENING_CFLAGS}")
message(STATUS "Using security hardening linker flags: ${HARDENING_LDFLAGS}")
message(STATUS "Using security hardening compiler definitions: ${HARDENING_DEFINITIONS}")

add_compile_options("${HARDENING_CFLAGS}")
add_link_options("${HARDENING_LDFLAGS}")
add_compile_definitions("${HARDENING_DEFINITIONS}")

# Build CUDA module by default if CUDA is available
if (BUILD_CUDA_MODULE)
    include(CloudViewerMakeCudaArchitectures)
    cloudViewer_make_cuda_architectures(CUDA_ARCHS)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS})

    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    enable_language(CUDA)

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

# Enumerate all third-party libraries which we need later
# This creates the necessary targets and sets the
# CloudViewer_3RDPARTY_*_TARGETS variables we use in cloudViewer_link_3rdparty_libraries
include(3rdparty/find_dependencies.cmake)

# Default debug suffix for libraries.
# Note: must put after include(3rdparty/find_dependencies.cmake)
set(CMAKE_DEBUG_POSTFIX "d")

# Add external libraries
include(CMakeExternalLibs)

# Contrib. libraries (mainly for I/O)
include(AllSupport)

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

# Internal libs used by both ErowCloudViewer and cloudViewer python package
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
add_subdirectory(eCV)

# Install CMake configuration files
install(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}:: DESTINATION ${CloudViewer_INSTALL_CMAKE_DIR})
export(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}::)

if (Python3_EXECUTABLE)
    # `make check-style` checks style for c++/cuda/python/ipynb files
    add_custom_target(check-style
            COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/util/check_style.py
            COMMENT "Python executable used for style check: ${Python3_EXECUTABLE}."
            )

    # `make apply-style` applies style for c++/cuda/python/ipynb files
    add_custom_target(apply-style
            COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/util/check_style.py --do_apply_style
            COMMENT "Python executable used for style check: ${Python3_EXECUTABLE}."
            )
endif ()

# `make check-cpp-style` checks style for c++/cuda files.
# This works outside of python virtualenv.
add_custom_target(check-cpp-style
        COMMAND ${CMAKE_COMMAND}
        -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        -DDO_APPLY_STYLE=OFF
        -P ${CMAKE_CURRENT_SOURCE_DIR}/util/check_cpp_style.cmake
        )

# `make apply-cpp-style` applies style for c++/cuda files.
# This works outside of python virtualenv.
add_custom_target(apply-cpp-style
        COMMAND ${CMAKE_COMMAND}
        -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        -DDO_APPLY_STYLE=ON
        -P ${CMAKE_CURRENT_SOURCE_DIR}/util/check_cpp_style.cmake
        )

include(CloudViewerPrintConfigurationSummary)
cloudViewer_print_configuration_summary()
