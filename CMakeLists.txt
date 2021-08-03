cmake_minimum_required(VERSION 3.19)
# If you're using Ubuntu 18.04, we suggest you install the latest CMake from the
# official repository https://apt.kitware.com/.  CMake 3.18+ is required to
# allow linking with OBJECT libraries, to prevent erroneous -gencode option
# deduplication with CUDA, and to simplify generator expressions for selecting
# compile flags and setting global hardened link flags.

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

# One shouldn't generate the BUILD project directly in the SOURCES folder!
if (${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    if (NOT SAME_BUILD_AND_SOURCE_FOLDER_WARNING_ALREADY_ISSUED)
        message(FATAL_ERROR "It is not advised to BUILD the binaries directly in the SOURCE folder!\n If you want to proceed with this option, just CONFIGURE the project once again")
        set(SAME_BUILD_AND_SOURCE_FOLDER_WARNING_ALREADY_ISSUED TRUE)
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
option(BUILD_GUI "Builds new GUI" ON)
option(WITH_OPENMP "Use OpenMP multi-threading" ON)
option(WITH_IPPICV "Use Intel Performance Primitives" ON)
option(ENABLE_HEADLESS_RENDERING "Use OSMesa for headless rendering" OFF)
option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" OFF)
option(GLIBCXX_USE_CXX11_ABI "Set -D_GLIBCXX_USE_CXX11_ABI=1" OFF)
option(BUILD_RPC_INTERFACE "Build the RPC interface" OFF)
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
option(USE_SYSTEM_QHULL "Use system pre-installed qhull" OFF)
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

# Release build options
option(DEVELOPER_BUILD "Add +commit_hash to the project version number" ON)
if (NOT DEVELOPER_BUILD)
    if (NOT BUILD_COMMON_CUDA_ARCHS)
        set(BUILD_COMMON_CUDA_ARCHS ON CACHE BOOL "Build for common CUDA GPUs (for release)" FORCE)
        message(WARNING "Setting BUILD_COMMON_CUDA_ARCHS=ON since DEVELOPER_BUILD is OFF.")
    endif ()
endif ()

# PCL ENGINE SUPPORT
option(BUILD_RECONSTRUCTION "Build the 3D Reconstruction" OFF)
option(USE_PCL_BACKEND "Check to use pcl library as backend" ON)

# other options
option(CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING "change all dynamic dll output path to main project output path for debugging!" ON)
option(BOOST_NO_CXX11 "if Boost is compiled without C++11 support (as it is often the case in OS packages) this must be enabled to avoid symbol conflicts (SCOPED_ENUM)." OFF)

# Default build type on single-config generators.
# For multi-config generators (e.g. Visual Studio), CMAKE_CONFIGURATION_TYPES
# will be set, and we don't specify a default CMAKE_BUILD_TYPE.
# https://blog.kitware.com/cmake-and-the-default-build-type/
if (NOT CMAKE_CONFIGURATION_TYPES)
    if (NOT CMAKE_BUILD_TYPE)
        message(STATUS "Setting build type to Release as none was specified.")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
        # Set the possible values of build type for cmake-gui.
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
    endif ()
    message(STATUS "CMAKE_BUILD_TYPE is set to ${CMAKE_BUILD_TYPE}.")
endif ()

# In ExternalProject_Add, if THIRD_PARTY_DOWNLOAD_DIR is specified, CMake will
# first try to look for the required files in THIRD_PARTY_DOWNLOAD_DIR, before
# downloading it from the internet. The files inside THIRD_PARTY_DOWNLOAD_DIR
# are prepared manually by the user. This is only supported by limited 3rd party
# libraries.
set(THIRD_PARTY_FROM "cmake options")
if (DEFINED ENV{THIRD_PARTY_DOWNLOAD_DIR} AND
        IS_DIRECTORY "$ENV{THIRD_PARTY_DOWNLOAD_DIR}" AND
        NOT THIRD_PARTY_DOWNLOAD_DIR)
    set(THIRD_PARTY_DOWNLOAD_DIR "$ENV{THIRD_PARTY_DOWNLOAD_DIR}")
    set(THIRD_PARTY_FROM "the environment")
endif ()
if (THIRD_PARTY_DOWNLOAD_DIR)
    message(STATUS "THIRD_PARTY_DOWNLOAD_DIR is set to "
            "${THIRD_PARTY_DOWNLOAD_DIR} from ${THIRD_PARTY_FROM}.")
else ()
    message(STATUS "THIRD_PARTY_DOWNLOAD_DIR is not specified, will download directly.")
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
file(STRINGS "libs/OpenGLEngine/version.txt" CLOUDVIEWER_VERSION_READ)
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
if (NOT DEFINED Python3_FIND_REGISTRY)
    # Only consider PATH variable on Windows by default
    set(Python3_FIND_REGISTRY NEVER)
endif ()
find_package(Python3 3.6...<3.10
        COMPONENTS Interpreter Development
        )
if (Python3_FOUND)
    # Setup PYTHON_EXECUTABLE for 3rdparty modules
    # which still use the deprecated find_package(PythonInterp)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE} CACHE STRING
            "Deprecated path to the Python executable (for 3rdparty only)" FORCE)
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
set(PROJECT_HOME "http://www.erow.cn")
set(PROJECT_DOCS "http://www.erow.cn/docs")
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
# -Wall -Wextra -Werror or /W4 /WX are enabled for CloudViewer code (not 3rd party)
if (MSVC)
    set(HARDENING_CFLAGS
            /sdl            # SDL Checks
            /GS             # Code Generation: Security Check
            /guard:cf       # Code Generation: Control Flow Guard
            CACHE STRING "Compiler flags for security hardening")
    set(HARDENING_LDFLAGS
            /INCREMENTAL:NO  # Disable incremental Linking
            /NXCOMPAT        # Data Execution Prevention: On by default in VS2019
            /DYNAMICBASE     # Randomized Base Address
            /HIGHENTROPYVA   #
            #/INTEGRITYCHECK # Signed binary: Disabled
            CACHE STRING "Linker flags for security hardening")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_definitions(_FORTIFY_SOURCE=2)     # Buffer overflow detection
    set(HARDENING_CFLAGS
            -fstack-protector               # Stack-based buffer overrun detection
            -Wformat -Wformat-security      # Format string vulnerability
            CACHE STRING "Compiler flags for security hardening")
    set(HARDENING_LDFLAGS
            -fsanitize=safe-stack       # Stack execution protection
            -Wl,-z,relro,-z,now         # Data relocation protection
            -pie                        # Position independent executable
            $<$<CONFIG:Release>:LINKER:-S>  # Exclude debug info
            CACHE STRING "Linker flags for security hardening")
    if (NOT BUILD_SHARED_LIBS AND NOT BUILD_PYTHON_MODULE)
        list(APPEND HARDENING_CFLAGS -fsanitize=safe-stack)   # Stack execution protection
        list(APPEND HARDENING_LDFLAGS -fsanitize=safe-stack)  # only static libraries supported
    endif ()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_compile_definitions(_FORTIFY_SOURCE=2)     # Buffer overflow detection
    set(HARDENING_CFLAGS
            -fstack-protector               # Stack-based buffer overrun detection
            -Wformat -Wformat-security      # Format string vulnerability
            CACHE STRING "Compiler flags for security hardening")
    set(HARDENING_LDFLAGS
            # -pie Position independent executable is default on macOSX 10.6+
            LINKER:-dead_strip    # Remove unreachable code
            $<$<CONFIG:Release>:LINKER:-S,-x>  # Exclude debug info, non-global symbols
            CACHE STRING "Linker flags for security hardening")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_definitions(_FORTIFY_SOURCE=2)     # Buffer overflow detection
    set(HARDENING_CFLAGS
            -fstack-protector-strong    # Stack-based buffer overrun detection
            -Wformat -Wformat-security  # Format string vulnerability
            CACHE STRING "Compiler flags for security hardening")
    set(HARDENING_LDFLAGS
            -Wl,-z,noexecstack   # Stack execution protection
            -Wl,-z,relro,-z,now  # Data relocation protection
            -pie                 # Position independent executable
            $<$<CONFIG:Release>:LINKER:--strip-debug>  # Exclude debug info
            CACHE STRING "Linker flags for security hardening")
else ()
    message(WARNING "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}. No security "
            "flags set")
endif ()
include(CheckCXXCompilerFlag)
foreach (FLAG ${HARDENING_CFLAGS})
    string(MAKE_C_IDENTIFIER "${FLAG}" FLAGRESULT)
    check_cxx_compiler_flag("${FLAG}" FLAG${FLAGRESULT})
    if (NOT FLAG${FLAGRESULT})
        list(REMOVE_ITEM HARDENING_CFLAGS ${FLAG})
        message(WARNING "Compiler does not support security option ${FLAG}")
    endif ()
endforeach ()
include(CheckLinkerFlag)
foreach (FLAG ${HARDENING_LDFLAGS})
    string(MAKE_C_IDENTIFIER "${FLAG}" FLAGRESULT)
    check_linker_flag(CXX "${FLAG}" FLAG${FLAGRESULT})              # cmake 3.18+
    if (NOT FLAG${FLAGRESULT})
        list(REMOVE_ITEM HARDENING_LDFLAGS ${FLAG})
        message(WARNING "Linker does not support security option ${FLAG}")
    endif ()
endforeach ()
message(STATUS "Using security hardening compiler flags: ${HARDENING_CFLAGS} and
linker flags: ${HARDENING_LDFLAGS}")
list(TRANSFORM HARDENING_LDFLAGS REPLACE "-pie"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>")
string(REPLACE ";" "," CUDA_HARDENING_CFLAGS "${HARDENING_CFLAGS}")
string(REPLACE ";" "," CUDA_HARDENING_LDFLAGS "${HARDENING_LDFLAGS}")
add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${CUDA_HARDENING_CFLAGS}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${HARDENING_CFLAGS}>"
)
add_link_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${CUDA_HARDENING_LDFLAGS}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${HARDENING_LDFLAGS}>"
)

# Build CUDA module by default if CUDA is available
if (BUILD_CUDA_MODULE)
    find_package(CUDAToolkit REQUIRED)

    if (BUILD_COMMON_CUDA_ARCHS)
        # Build with all supported architectures for previous 2 generations and
        # M0 (minor=0) architectures for previous generations (inluding
        # deprecated). Note that cubin for M0 runs on GPUs with architecture Mx.
        # This is a tradeoff between binary size / build time and runtime on
        # older architectures.
        # See https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#building-for-maximum-compatibility.
        # https://docs.nvidia.com/cuda/ampere-compatibility-guide/index.html#application-compatibility-on-ampere
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.1")
            set(CMAKE_CUDA_ARCHITECTURES 35-real 50-real 60-real 70-real 72-real 75-real 80-real 86)
        elseif (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
            set(CMAKE_CUDA_ARCHITECTURES 35-real 50-real 60-real 70-real 72-real 75-real 80)
        else ()
            set(CMAKE_CUDA_ARCHITECTURES 30-real 50-real 60-real 70-real 72-real 75)
        endif ()
    else ()
        if (CMAKE_CUDA_ARCHITECTURES)
            message(STATUS "Building with user-provided architectures")
        else ()
            file(WRITE
                    "${CMAKE_CURRENT_BINARY_DIR}/cuda_architectures.c"
                    "
                #include <stdio.h>
                #include <cuda_runtime_api.h>
                int main() {
                    int n;
                    if (cudaGetDeviceCount(&n) == cudaSuccess) {
                        for (int i = 0; i < n; ++i) {
                            int major, minor;
                            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                                    i) == cudaSuccess &&
                                cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                                    i) == cudaSuccess) {
                                if (i > 0) {
                                    printf(\";\");
                                }
                                printf(\"%d%d-real\", major, minor);
                            }
                        }
                    }
                    return 0;
                }
                ")
            try_run(
                    DETECTION_RETURN_VALUE DETECTION_COMPILED
                    "${CMAKE_CURRENT_BINARY_DIR}"
                    "${CMAKE_CURRENT_BINARY_DIR}/cuda_architectures.c"
                    LINK_LIBRARIES CUDA::cudart
                    RUN_OUTPUT_VARIABLE DETECTED_ARCHITECTURES)
            if (DETECTED_ARCHITECTURES)
                message(STATUS "Building with detected architectures.")
                set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_ARCHITECTURES})
            else ()
                message(STATUS "Failed to detect architectures. Falling back to CMake's default architectures.")
            endif ()
        endif ()
    endif ()

    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    enable_language(CUDA)

    if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "10.1")
        message(FATAL_ERROR "CUDA 10.0 and older are not supported. Please upgrade to CUDA 10.1 or newer.")
    endif ()
endif ()

# OS specific settings
if (WIN32)
    # Windows defaults to hidden symbol visibility, override that
    # TODO: It would be better to explictly export symbols.
    #       Then, we could use -fvisibility=hidden for Linux as well
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
	if (STATIC_WINDOWS_RUNTIME)
        # This needs cmake_policy(SET CMP0091 NEW)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded<<CONFIG:Debug>:Debug>")
    endif()
    # if (STATIC_WINDOWS_RUNTIME)
        # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # else ()
        # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    # endif ()
endif ()

# Downloads files from https://github.com/Asher-1/cloudViewer_downloads to
# Open3D/examples/test_data/cloudViewer_downloads.
#add_custom_target(cloudViewer_downloads
#        COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/examples/test_data/cloudViewer_downloads.cmake
#        COMMENT ""
#        )

include(${PROJECT_SOURCE_DIR}/examples/cloudViewer_downloads.cmake)

# Folder view for project files
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Convenience function to link against all third-party libraries
# We need this because we create a lot of object libraries to assemble
# the main ErowCloudViewer library
function(cloudViewer_link_3rdparty_libraries target)
    target_link_libraries(${target} PRIVATE ${CloudViewer_3RDPARTY_PRIVATE_TARGETS})
    target_link_libraries(${target} PUBLIC ${CloudViewer_3RDPARTY_PUBLIC_TARGETS})
    include_directories(${3RDPARTY_INCLUDE_DIRS})
    target_link_directories(${target} PUBLIC ${3RDPARTY_LIBRARY_DIRS})
    target_link_libraries(${target} PUBLIC ${3RDPARTY_LIBRARIES})

    foreach (dep IN LISTS CloudViewer_3RDPARTY_HEADER_TARGETS)
        if (TARGET ${dep})
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
            if (inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${inc})
            endif ()
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
            if (inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${inc})
            endif ()
            get_property(def TARGET ${dep} PROPERTY INTERFACE_COMPILE_DEFINITIONS)
            if (def)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ${def})
            endif ()
        endif ()
    endforeach ()
endfunction()

# Check if the compiler defines the _GLIBCXX_USE_CXX11_ABI macro
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("#include <cxxabi.h>
int main() { return _GLIBCXX_USE_CXX11_ABI; }" HAS_GLIBCXX_USE_CXX11_ABI)

# Convenience function to set important target properties
function(cloudViewer_set_global_properties target)
    # Tell CMake we want a compiler that supports C++14 features
    target_compile_features(${target} PUBLIC cxx_std_14)

    # Colorize GCC/Clang terminal outputs
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>)
    endif ()

    target_include_directories(${target} PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp>
            $<INSTALL_INTERFACE:${Open3D_INSTALL_INCLUDE_DIR}>
            )

    # Propagate build configuration into source code
    if (BUILD_CUDA_MODULE)
        target_compile_definitions(${target} PRIVATE BUILD_CUDA_MODULE)
        if (BUILD_CACHED_CUDA_MANAGER)
            target_compile_definitions(${target} PRIVATE BUILD_CACHED_CUDA_MANAGER)
        endif ()
    endif ()
    if (BUILD_GUI)
        target_compile_definitions(${target} PRIVATE BUILD_GUI)
    endif ()
    if (ENABLE_HEADLESS_RENDERING)
        target_compile_definitions(${target} PRIVATE HEADLESS_RENDERING)
    endif ()
    if (BUILD_RPC_INTERFACE)
        target_compile_definitions(${target} PRIVATE BUILD_RPC_INTERFACE ZMQ_STATIC)
    endif ()
    if (BUILD_AZURE_KINECT)
        target_compile_definitions(${target} PRIVATE BUILD_AZURE_KINECT)
    endif ()
    if (BUILD_LIBREALSENSE)
        target_compile_definitions(${target} PRIVATE BUILD_LIBREALSENSE)
    endif ()
    if (BUILD_WEBRTC)
        target_compile_definitions(${target} PRIVATE BUILD_WEBRTC)
    endif ()
    if (USE_BLAS)
        target_compile_definitions(${target} PRIVATE USE_BLAS)
    endif ()
    if (WITH_IPPICV)
        target_compile_definitions(${target} PRIVATE WITH_IPPICV)
    endif ()
    if (WITH_FAISS)
        target_compile_definitions(${target} PRIVATE WITH_FAISS)
    endif ()
    if (GLIBCXX_USE_CXX11_ABI)
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
    else ()
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
    endif ()

    if (NOT WITH_OPENMP)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>")
    endif ()
    if (WIN32)
        target_compile_definitions(${target} PRIVATE
                WINDOWS
                _CRT_SECURE_NO_DEPRECATE
                _CRT_NONSTDC_NO_DEPRECATE
                _SCL_SECURE_NO_WARNINGS
                )
        if (MSVC)
            target_compile_definitions(${target} PRIVATE NOMINMAX _USE_MATH_DEFINES _ENABLE_EXTENDED_ALIGNED_STORAGE)
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
            target_compile_options(${target} PRIVATE /wd4522 /wd4190 /wd4819 /wd4717 /Wv:18 /WX-)
            # Multi-thread compile, two ways to enable
            # Option 1, at build time: cmake --build . --parallel %NUMBER_OF_PROCESSORS%
            # https://stackoverflow.com/questions/36633074/set-the-number-of-threads-in-a-cmake-build
            # Option 2, at configure time: add /MP flag, no need to use Option 1
            # https://docs.microsoft.com/en-us/cpp/build/reference/mp-build-with-multiple-processes?view=vs-2019
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP>)
            if (BUILD_GUI)
                # GLEW and Open3D make direct OpenGL calls and link to opengl32.lib;
                # Filament needs to link through bluegl.lib.
                # See https://github.com/google/filament/issues/652
                target_link_options(${target} PRIVATE /force:multiple)
            endif ()
            # The examples' .pdb files use up a lot of space and cause us to run
            # out of space on Github Actions. Compressing gives us another couple of GB.
            target_link_options(${target} PRIVATE /pdbcompress)
        endif ()
    elseif (APPLE)
        target_compile_definitions(${target} PRIVATE UNIX APPLE)
    elseif (UNIX)
        target_compile_definitions(${target} PRIVATE UNIX)
    endif ()
    if (LINUX_AARCH64)
        target_compile_definitions(${target} PRIVATE LINUX_AARCH64)
    endif ()
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>")

    # TBB static version is used
    # See: https://github.com/wjakob/tbb/commit/615d690c165d68088c32b6756c430261b309b79c
    target_compile_definitions(${target} PRIVATE __TBB_LIB_NAME=tbb_static)

    # Download test data files from cloudViewer_downloads repo.
    add_dependencies(${target} ext_cloudViewer_downloads)

    # Strip unnecessary sections of the binary on Linux/macOS for Release builds
    # (from pybind11)
    # macOS: -x: strip local symbols
    # Linux: defaults
    if (UNIX AND CMAKE_STRIP)
        get_target_property(target_type ${target} TYPE)
        if (target_type MATCHES
                MODULE_LIBRARY|SHARED_LIBRARY|EXECUTABLE)
            add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND
                    $<IF:$<CONFIG:Release>,${CMAKE_STRIP},true>
                    $<$<PLATFORM_ID:Darwin>:-x> $<TARGET_FILE:${target}>
                    COMMAND_EXPAND_LISTS)
        endif ()
    endif ()

endfunction()

# recursively parse and return the entire directory tree.
# the result is placed in output
function(Directories root output)
    set(data "")
    list(APPEND data ${root})
    file(GLOB_RECURSE children LIST_DIRECTORIES true "${root}/*")
    list(SORT children)
    foreach (child ${children})
        if (IS_DIRECTORY ${child})
            list(APPEND data ${child})
        endif ()
    endforeach ()
    set(${output} ${data} PARENT_SCOPE)
endfunction()

function(cloudViewer_show_and_abort_on_warning trgt)
    set(DISABLE_MSVC_WARNINGS
            /Wv:18         # ignore warnings introduced in Visual Studio 2015 and later.
            /wd4201        # non-standard extension nameless struct (filament includes)
            /wd4310        # cast truncates const value (filament)
            /wd4505        # unreferenced local function has been removed (dirent)
            /wd4127        # conditional expression is const (Eigen)
            /wd4146        # unary minus operator applied to unsigned type, result still unsigned (UnaryEWCPU)
            /wd4189        # local variable is initialized but not referenced (PoissonRecon)
            /wd4324        # structure was padded due to alignment specifier (qhull)
            /wd4706        # assignment within conditional expression (fileIO, ...)
            /wd4100        # unreferenced parameter (many places in Open3D code)
            /wd4702        # unreachable code (many places in Open3D code)
            /wd4244        # implicit data type conversion (many places in Open3D code)
            /wd4245        # signed/unsigned mismatch (visualization, PoissonRecon, ...)
            /wd4267        # conversion from size_t to smaller type (FixedRadiusSearchCUDA, tests)
            /wd4305        # conversion to smaller type in initialization or constructor argument (examples, tests)
            )
    set(DISABLE_GNU_CLANG_INTEL_WARNINGS
            -Wno-unused-parameter               # (many places in Open3D code)
            )

    if (BUILD_CUDA_MODULE)
        # General NVCC flags
        set(DISABLE_NVCC_WARNINGS
                2809           # ignoring return value from routine declared with "nodiscard" attribute (cub)
                )
        string(REPLACE ";" "," DISABLE_NVCC_WARNINGS "${DISABLE_NVCC_WARNINGS}")

        set(CUDA_FLAGS "--Werror cross-execution-space-call,deprecated-declarations")
        if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "10.2")
            string(APPEND CUDA_FLAGS " --Werror all-warnings")
        endif ()
        if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
            string(APPEND CUDA_FLAGS " --Werror ext-lambda-captures-this")
        endif ()
        string(APPEND CUDA_FLAGS " --expt-relaxed-constexpr")
        if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.2")
            string(APPEND CUDA_FLAGS " --diag-suppress ${DISABLE_NVCC_WARNINGS}")
        else ()
            string(APPEND CUDA_FLAGS " -Xcudafe --diag_suppress=[${DISABLE_NVCC_WARNINGS}]")
        endif ()

        # Host compiler flags
        if (MSVC)
            set(CUDA_DISABLE_MSVC_WARNINGS ${DISABLE_MSVC_WARNINGS})
            if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.2")
                list(APPEND CUDA_DISABLE_MSVC_WARNINGS $<$<CONFIG:Debug>:/wd4700>) # uninitialized local variable used (thrust)
            endif ()
            if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.1")
                list(APPEND CUDA_DISABLE_MSVC_WARNINGS /wd4515) # namespace uses itself (thrust)
            endif ()
            string(REPLACE ";" "," CUDA_DISABLE_MSVC_WARNINGS "${CUDA_DISABLE_MSVC_WARNINGS}")

            string(APPEND CUDA_FLAGS " -Xcompiler /W4,/WX,${CUDA_DISABLE_MSVC_WARNINGS}")
        else ()
            # reorder breaks builds on Windows, so only enable for other platforms
            string(APPEND CUDA_FLAGS " --Werror reorder")

            set(CUDA_DISABLE_GNU_CLANG_INTEL_WARNINGS ${DISABLE_GNU_CLANG_INTEL_WARNINGS})
            string(REPLACE ";" "," CUDA_DISABLE_GNU_CLANG_INTEL_WARNINGS "${CUDA_DISABLE_GNU_CLANG_INTEL_WARNINGS}")

            string(APPEND CUDA_FLAGS " -Xcompiler -Wall,-Wextra,-Werror,${CUDA_DISABLE_GNU_CLANG_INTEL_WARNINGS}")
        endif ()
    else ()
        set(CUDA_FLAGS "")
    endif ()

    target_compile_options(${trgt} PRIVATE
            $<$<COMPILE_LANG_AND_ID:C,MSVC>:/W4 /WX- ${DISABLE_MSVC_WARNINGS}>
            $<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang,Intel>:-Wall -Wextra ${DISABLE_GNU_CLANG_INTEL_WARNINGS}>
            $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4 /WX- ${DISABLE_MSVC_WARNINGS}>
            $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,Intel>:-Wall -Wextra ${DISABLE_GNU_CLANG_INTEL_WARNINGS}>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:${CUDA_FLAGS}>
            )
endfunction()

function(CopyDLLToDirectory trgt)
    add_custom_command(TARGET ${trgt}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${trgt}>" "${CLOUDVIEWER_OUTPUT_DIRECTORY}/"
            VERBATIM)
    if (${BUILD_GUI})
        # Install dependence lib files
        if (WIN32)
            foreach (filename "$<TARGET_FILE:${trgt}>")
                cloudViewer_install_ext(FILES ${filename} "${CMAKE_INSTALL_PREFIX}/bin/${COLOUDVIEWER_APP_DIR_NAME}" "")
            endforeach ()
        endif ()
    endif ()
endfunction()

macro(ADD_SOURCE_GROUP MODULE_NAME)
    file(GLOB MODULE_HEADER_FILES "${MODULE_NAME}/*.h")
    source_group("Header Files\\${MODULE_NAME}" FILES ${MODULE_HEADER_FILES})
    file(GLOB MODULE_SOURCE_FILES "${MODULE_NAME}/*.cpp")
    source_group("Source Files\\${MODULE_NAME}" FILES ${MODULE_SOURCE_FILES})
    file(GLOB MODULE_SHADER_FILES "${MODULE_NAME}/*.glsl")
    source_group("Source Files\\Shader\\GLSL" FILES ${MODULE_SHADER_FILES})
    file(GLOB MODULE_MATERIAL_FILES "${MODULE_NAME}/*.mat")
    source_group("Source Files\\Material" FILES ${MODULE_MATERIAL_FILES})
endmacro(ADD_SOURCE_GROUP)

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

# print configurations info
function(cloudviewer_aligned_print first second)
    string(LENGTH "${first}" first_len)
    math(EXPR star_len "40 - ${first_len}")
    # CMake 3.15+ has string(REPEAT), but this is backwards compatible
    string(SUBSTRING "........................................" 0 ${star_len} stars)
    message(STATUS "  ${first} ${stars} ${second}")
endfunction()

message(STATUS "========================================================================")
message(STATUS "CloudViewer ${PROJECT_VERSION} Configuration Summary")
message(STATUS "========================================================================")
message(STATUS "Enabled Features:")
cloudviewer_aligned_print("OpenMP" "${WITH_OPENMP}")
cloudviewer_aligned_print("SIMD" "${USE_SIMD}")
cloudviewer_aligned_print("Headless Rendering" "${ENABLE_HEADLESS_RENDERING}")
cloudviewer_aligned_print("Azure Kinect Support" "${BUILD_AZURE_KINECT}")
cloudviewer_aligned_print("Intel RealSense Support" "${BUILD_LIBREALSENSE}")
cloudviewer_aligned_print("3D Reconstruction Support" "${BUILD_RECONSTRUCTION}")
cloudviewer_aligned_print("CUDA Support" "${BUILD_CUDA_MODULE}")
cloudviewer_aligned_print("Build GUI" "${BUILD_GUI}")
cloudviewer_aligned_print("Build WebRTC visualizer" "${BUILD_WEBRTC}")
cloudviewer_aligned_print("Build Shared Library" "${BUILD_SHARED_LIBS}")
if (WIN32)
    cloudviewer_aligned_print("Use Windows Static Runtime" "${STATIC_WINDOWS_RUNTIME}")
endif ()
cloudviewer_aligned_print("Build Unit Tests" "${BUILD_UNIT_TESTS}")
cloudviewer_aligned_print("Build Examples" "${BUILD_EXAMPLES}")
cloudviewer_aligned_print("Build Python Module" "${BUILD_PYTHON_MODULE}")
cloudviewer_aligned_print("Build Jupyter Extension" "${BUILD_JUPYTER_EXTENSION}")
cloudviewer_aligned_print("Build Tensorflow Ops" "${BUILD_TENSORFLOW_OPS}")
cloudviewer_aligned_print("Build Pytorch Ops" "${BUILD_PYTORCH_OPS}")
if (BUILD_PYTORCH_OPS AND BUILD_CUDA_MODULE AND
        CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
    message(WARNING
            "--------------------------------------------------------------------------------\n"
            "                                                                                \n"
            " You are compiling PyTorch ops with CUDA 11. This configuration may have        \n"
            " stability issues. See https://github.com/intel-isl/Open3D/issues/3324 and      \n"
            " https://github.com/pytorch/pytorch/issues/52663 for more information on this   \n"
            " problem.                                                                       \n"
            "                                                                                \n"
            " We recommend to compile PyTorch from source with compile flags                 \n"
            "   '-Xcompiler -fno-gnu-unique'                                                 \n"
            "                                                                                \n"
            " or use the PyTorch wheels at                                                   \n"
            "   https://github.com/intel-isl/open3d_downloads/releases/tag/torch1.8.1        \n"
            "                                                                                \n"
            "--------------------------------------------------------------------------------\n"
            )
endif ()
cloudviewer_aligned_print("Build Benchmarks" "${BUILD_BENCHMARKS}")
cloudviewer_aligned_print("Bundle CloudViewer-ML" "${BUNDLE_CLOUDVIEWER_ML}")
cloudviewer_aligned_print("Build RPC interface" "${BUILD_RPC_INTERFACE}")
if (GLIBCXX_USE_CXX11_ABI)
    set(usage "1")
else ()
    set(usage "0")
endif ()
cloudviewer_aligned_print("Force GLIBCXX_USE_CXX11_ABI=" "${usage}")
message(STATUS "========================================================================")
message(STATUS "Third-Party Dependencies:")
set(3RDPARTY_DEPENDENCIES
    Eigen3
    faiss
    filament
    fmt
    GLEW
    GLFW
    imgui
    ippicv
    JPEG
    jsoncpp
    liblzf
    OpenGL
    PNG
    qhullcpp
    librealsense
    tinyfiledialogs
    TinyGLTF
    tinyobjloader
    WebRTC
        )

foreach(dep IN LISTS 3RDPARTY_DEPENDENCIES)
    string(TOLOWER "${dep}" dep_lower)
    string(TOUPPER "${dep}" dep_upper)
    if(TARGET 3rdparty_${dep_lower})
        if(NOT USE_SYSTEM_${dep_upper})
            cloudviewer_aligned_print("${dep}" "yes (build from source)")
        else()
            if(3rdparty_${dep_lower}_VERSION)
                cloudviewer_aligned_print("${dep}" "yes (v${3rdparty_${dep_lower}_VERSION})")
            else()
                cloudviewer_aligned_print("${dep}" "yes")
            endif()
        endif()
    else()
        cloudviewer_aligned_print("${dep}" "no")
    endif()
endforeach()
message(STATUS "========================================================================")
