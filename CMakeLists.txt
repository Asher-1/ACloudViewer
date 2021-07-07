cmake_minimum_required(VERSION 3.18)
# If you're using Ubuntu 18.04, we suggest you install the latest CMake from the
# official repository https://apt.kitware.com/.  CMake 3.18+ is required to
# allow linking with OBJECT libraries, to prevent erroneous -gencode option
# deduplication with CUDA, and to simplify generator expressions for selecting
# compile flags and setting global hardened link flags.

if (APPLE)
set (CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING
    "Minimum OS X deployment version" FORCE)
endif()

# One shouldn't generate the BUILD project directly in the SOURCES folder!
if ( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )
	if ( NOT SAME_BUILD_AND_SOURCE_FOLDER_WARNING_ALREADY_ISSUED )
		message(FATAL_ERROR "It is not advised to BUILD the binaries directly in the SOURCE folder!\n If you want to proceed with this option, just CONFIGURE the project once again" )
		set( SAME_BUILD_AND_SOURCE_FOLDER_WARNING_ALREADY_ISSUED TRUE )
	endif()
endif()

# ErowCloudViewer build options
option(BUILD_SHARED_LIBS          "Build shared libraries"                   			    OFF)
option(BUILD_EXAMPLES             "Build CloudViewer example programs"        			    ON )
option(BUILD_UNIT_TESTS           "Build CloudViewer unit tests"                        	OFF)
option(BUILD_BENCHMARKS           "Build the micro benchmarks"                          	OFF)
option(BUILD_PYTHON_MODULE        "Build the python module"                             	ON )
option(BUILD_RECONSTRUCTION       "Build the 3D Reconstruction"                             OFF)
option(BUILD_CUDA_MODULE          "Build the CUDA module"                               	OFF)
option(BUILD_CACHED_CUDA_MANAGER  "Build the cached CUDA memory manager"                	ON )
option(BUILD_GUI                  "Builds new GUI"                                      	ON )
option(BUILD_JUPYTER_EXTENSION    "Enable Jupyter support for CloudViewer"              	OFF)
option(ENABLE_HEADLESS_RENDERING  "Use OSMesa for headless rendering"                   	OFF)
option(STATIC_WINDOWS_RUNTIME     "Use static (MT/MTd) Windows runtime"                 	OFF)
option(GLIBCXX_USE_CXX11_ABI      "Set -D_GLIBCXX_USE_CXX11_ABI=1"                      	OFF)
option(BUILD_RPC_INTERFACE        "Build the RPC interface"                             	OFF)
option(BUILD_FILAMENT_FROM_SOURCE "Build filament from source"                          	OFF)
# 3rd-party build options
option(USE_SIMD                   "Use Single Instruction Multiple Data speed optimization" OFF)
option(USE_BLAS                   "Use BLAS/LAPACK instead of MKL"                      	OFF)
option(USE_SYSTEM_EIGEN3          "Use system pre-installed eigen3"                     	OFF)
option(USE_SYSTEM_FLANN           "Use system pre-installed flann"                      	OFF)
option(USE_SYSTEM_FMT             "Use system pre-installed fmt"                        	OFF)
option(USE_SYSTEM_GLEW            "Use system pre-installed glew"                       	OFF)
option(USE_SYSTEM_GLFW            "Use system pre-installed glfw"                       	OFF)
option(USE_SYSTEM_GOOGLETEST      "Use system pre-installed googletest"                 	OFF)
option(USE_SYSTEM_IMGUI           "Use system pre-installed imgui"                      	OFF)
option(USE_SYSTEM_JPEG            "Use system pre-installed jpeg"                       	OFF)
option(USE_SYSTEM_LIBLZF          "Use system pre-installed liblzf"                     	OFF)
option(USE_SYSTEM_PNG             "Use system pre-installed png"                        	OFF)
option(USE_SYSTEM_PYBIND11        "Use system pre-installed pybind11"                   	OFF)
option(USE_SYSTEM_QHULL           "Use system pre-installed qhull"                      	OFF)
option(USE_SYSTEM_TINYGLTF        "Use system pre-installed tinygltf"                   	OFF)
option(USE_SYSTEM_TINYOBJLOADER   "Use system pre-installed tinyobjloader"              	OFF)
# PCL ENGINE SUPPORT
option(USE_PCL_BACKEND           "Check to use pcl library as backend" 				        ON )
# Mac options
option(PREFER_OSX_HOMEBREW        "Prefer Homebrew libs over frameworks"                	ON )
# Sensor options
option(BUILD_LIBREALSENSE         "Build support for Intel RealSense camera"            	OFF)
option(USE_SYSTEM_LIBREALSENSE    "Use system pre-installed librealsense"               	OFF)
option(BUILD_AZURE_KINECT         "Build support for Azure Kinect sensor"               	OFF)
# ML library options
option(BUILD_TENSORFLOW_OPS       "Build ops for Tensorflow"                            	OFF)
option(BUILD_PYTORCH_OPS          "Build ops for Pytorch"                               	OFF)
option(BUNDLE_CLOUDVIEWER_ML      "Includes the CloudViewer-ML repo in the wheel"       	OFF)
option(DEVELOPER_BUILD            "Add +commit_hash to the project version number"      	ON )

# other options
option(WITH_FAISS                 "Enable Faiss"                                        	ON )
option(WITH_OPENMP                "Use OpenMP multi-threading"                          	ON )
option(WITH_IPPICV                "Use Intel Performance Primitives"                    	ON )
option(CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING "change all dynamic dll output path to main project output path for debugging!" ON )
option(BOOST_NO_CXX11 "if Boost is compiled without C++11 support (as it is often the case in OS packages) this must be enabled to avoid symbol conflicts (SCOPED_ENUM)." OFF)
# In ExternalProject_Add, if THIRD_PARTY_DOWNLOAD_DIR is specified, CMake will
# first try to look for the required files in THIRD_PARTY_DOWNLOAD_DIR, before
# downloading it from the internet. The files inside THIRD_PARTY_DOWNLOAD_DIR
# are prepared manually by the user. This is only supported by limited 3rd party
# libraries.

set(THIRD_PARTY_FROM "cmake options")
if(DEFINED ENV{THIRD_PARTY_DOWNLOAD_DIR} AND
        IS_DIRECTORY "$ENV{THIRD_PARTY_DOWNLOAD_DIR}" AND
        NOT THIRD_PARTY_DOWNLOAD_DIR)
    set(THIRD_PARTY_DOWNLOAD_DIR "$ENV{THIRD_PARTY_DOWNLOAD_DIR}")
    set(THIRD_PARTY_FROM "the environment")
endif()
if(THIRD_PARTY_DOWNLOAD_DIR)
    message(STATUS "THIRD_PARTY_DOWNLOAD_DIR is set to "
    "${THIRD_PARTY_DOWNLOAD_DIR} from ${THIRD_PARTY_FROM}.")
else()
    message(STATUS "THIRD_PARTY_DOWNLOAD_DIR is not specified, will download directly.")
endif()

set(FILAMENT_PRECOMPILED_ROOT "" CACHE PATH "Path to precompiled Filament library (used if BUILD_FILAMENT_FROM_SOURCE=OFF)")

if (PREFER_OSX_HOMEBREW)
    set(CMAKE_FIND_FRAMEWORK LAST)
    set(CMAKE_FIND_APPBUNDLE LAST)
endif()

# Cache variables for specifying the GPU architectures
set(CUDA_ARCH "Auto" CACHE STRING "Selects GPU architectures for code generation, \
one of (Auto|BasicPTX|User). Set to 'User' to set a custom list of architectures" )
set_property(CACHE CUDA_ARCH PROPERTY STRINGS Auto BasicPTX User)
set(CUDA_ARCH_USER "" CACHE STRING "User defined list of GPU architectures, e.g. 3.5 5.0+PTX Turing" )

# Default build type, default to RelWithDebInfo
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No CMAKE_BUILD_TYPE specified, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()


# Set OpenGL policy
if (POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
    set(CMP0072_VALUE NEW)
    if (NOT USE_SYSTEM_GLFW)
        cmake_policy(SET CMP0072 OLD)
        set(CMP0072_VALUE OLD)
    endif()
endif()

# Set policy for hidden symbol visibility
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

# Set policy to enable MSVC_RUNTIME_LIBRARY property.
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()

# Catch a few incompatible build options
if (APPLE AND ENABLE_HEADLESS_RENDERING)
    message(STATUS "Headless rendering is not supported on Mac OS")
    set(ENABLE_HEADLESS_RENDERING OFF)
endif()
if(ENABLE_HEADLESS_RENDERING AND BUILD_GUI)
    message(STATUS "Headless rendering disables the CloudViewer GUI")
    set(BUILD_GUI OFF)
endif()
if(ENABLE_HEADLESS_RENDERING AND (USE_SYSTEM_GLEW OR USE_SYSTEM_GLFW))
    message(STATUS "Headless rendering requires customized GLEW and GLFW builds")
    set(USE_SYSTEM_GLEW OFF)
    set(USE_SYSTEM_GLFW OFF)
endif()
if(BUNDLE_CLOUDVIEWER_ML AND NOT (BUILD_TENSORFLOW_OPS OR BUILD_PYTORCH_OPS))
    message(SEND_ERROR "3DML depends on Tensorflow or PyTorch Ops. Enable them with -DBUILD_TENSORFLOW_OPS=ON or -DBUILD_PYTORCH_OPS=ON")
endif()

# central location for specifying the CloudViewer version
file(STRINGS "libs/OpenGLEngine/version.txt" CLOUDVIEWER_VERSION_READ)
foreach(ver ${CLOUDVIEWER_VERSION_READ})
  if (ver MATCHES "CLOUDVIEWER_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
    set(CLOUDVIEWER_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
  endif()
endforeach()
set(CLOUDVIEWER_VERSION_DEVHASH "")
if(DEVELOPER_BUILD)
    execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" log --pretty=format:%h -n 1
        OUTPUT_VARIABLE GIT_REV)
    if (GIT_REV)
        set(CLOUDVIEWER_VERSION_DEVHASH "+${GIT_REV}")
    endif()
endif()
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

# Default debug suffix for libraries.
set( CMAKE_DEBUG_POSTFIX "d" )
message(STATUS "ErowCloudViewerProjects ${CLOUDVIEWER_VERSION_FULL}")

if(UNIX AND NOT APPLE)
    execute_process(COMMAND uname -p
        OUTPUT_VARIABLE PROCESSOR_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(LINUX_AARCH64 TRUE)
    endif()
endif()

# Set BLAS/LAPACK library preference.
if(LINUX_AARCH64)
    if(NOT USE_BLAS)
        set(USE_BLAS ON)
        message(WARNING "ARM CPU detected, setting USE_BLAS=ON.")
    endif()
    if(BUILD_GUI AND (NOT BUILD_FILAMENT_FROM_SOURCE))
        set(BUILD_FILAMENT_FROM_SOURCE ON)
        message(WARNING "ARM CPU detected, setting BUILD_FILAMENT_FROM_SOURCE=ON.")
    endif()
endif()

# CMake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/3rdparty/CMake)

# Add our cmake module path so we don't need relative paths for these
list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" )
set (ECV_CMAKE_SCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# setup PYTHON_EXECUTABLE if not set
find_package(PythonExecutable REQUIRED) # invokes the module in 3rdparty/CMake

# PyPI package name controls specifies the repository name on PyPI. The default
# name is "cloudViewer". In the past, for historical reasons, we've used the
# following names for PyPI, while they are now deprecated:
# - cloudViewer-python
# - py3d
# - cloudViewer-original
# - cloudViewer-official
# - cloud-viewer
if(NOT DEFINED PYPI_PACKAGE_NAME)
    set(PYPI_PACKAGE_NAME "cloudViewer")
endif()

# set additional info
set(PROJECT_EMAIL   "dahailu@foxmail.com")
set(PROJECT_HOME    "http://www.erow.cn")
set(PROJECT_DOCS    "http://www.erow.cn/docs")
set(PROJECT_CODE    "https://github.com/Asher-1/ErowCloudViewer")
set(PROJECT_ISSUES  "https://github.com/Asher-1/ErowCloudViewer/issues")

include( CMakePolicies )
include( CMakeSetCompilerOptions )
include( DeployQt )
# Load advanced scripts
include( CMakeInclude )
include( Install )

# Set installation paths
if(UNIX OR CYGWIN)
    include(GNUInstallDirs)
    set(CloudViewer_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
    set(CloudViewer_INSTALL_BIN_DIR "${CMAKE_INSTALL_BINDIR}")
    set(CloudViewer_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}")
    set(CloudViewer_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
else()
    set(CloudViewer_INSTALL_INCLUDE_DIR include)
    set(CloudViewer_INSTALL_BIN_DIR bin)
    set(CloudViewer_INSTALL_LIB_DIR lib)
    set(CloudViewer_INSTALL_CMAKE_DIR "share/CloudViewer")
endif()

# RPATH Linux/Unix: (dynamic) libs are put in a subdir of prefix/lib,
# since they are only used by qCC/ccViewer
if( UNIX AND NOT APPLE )
	include( GNUInstallDirs )
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}/erowcloudviewer")
endif()

# Define target folders
# (now that ccViewer can have its own plugins, ErowCloudViewer and CloudViewer must fall in separate folders!
if(WIN32 OR APPLE)
	set( EROWCLOUDVIEWER_DEST_FOLDER "bin/ErowCloudViewer" )
else()
	set( EROWCLOUDVIEWER_DEST_FOLDER bin )
endif()

# set default install folders
if( WIN32 )
    # (on Windows - msvc generator - the '_debug' suffix is automatically added for debug configurations)
    set( INSTALL_DESTINATIONS ${EROWCLOUDVIEWER_DEST_FOLDER} )
elseif( UNIX AND NOT APPLE )
    set( INSTALL_DESTINATIONS ${CMAKE_INSTALL_PREFIX} )
endif()

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
# -Wall -Wextra -Werror or /W4 /WX are enabled for ErowCloudViewer code (not 3rd party)
if (MSVC)
    set(HARDENING_CFLAGS
        /sdl            # SDL Checks
        /GS             # Code Generation: Security Check
        /guard:cf       # Code Generation: Control Flow Guard
        CACHE STRING "Compiler flags for security hardening")
    set(HARDENING_LDFLAGS
        /INCREMENTAL:NO  # Disable incremental Linking
        #/NXCOMPAT       # Data Execution Prevention: On by default in VS2019
        /DYNAMICBASE     # Randomized Base Address
        /HIGHENTROPYVA   #
        #/INTEGRITYCHECK # Signed binary: Disabled
        CACHE STRING "Linker flags for security hardening")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(HARDENING_CFLAGS
        -fsanitize=safe-stack           # Stack execution protection
        -fstack-protector               # Stack-based buffer overrun detection
        -fpie -fPIC                     # Position independent execution
        -D_FORTIFY_SOURCE=2
        -Wformat -Wformat-security      # Format string vulnerability
        -fsanitize=cfi                  # Control flow integrity
        CACHE STRING "Compiler flags for security hardening")
    set(HARDENING_LDFLAGS
        -Wl,-z,relro,-z,now         # Data relocation protection
        -pie                        # Position independent execution
        CACHE STRING "Linker flags for security hardening")
    if(NOT DEVELOPER_BUILD)     # Strip debug symbols
        list(APPEND HARDENING_CFLAGS -O2)
        list(APPEND HARDENING_LDFLAGS -S)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(HARDENING_CFLAGS
        -fstack-protector               # Stack-based buffer overrun detection
        -fpie -fPIC                     # Position independent execution
        -D_FORTIFY_SOURCE=2
        -Wformat -Wformat-security      # Format string vulnerability
        CACHE STRING "Compiler flags for security hardening")
    set(HARDENING_LDFLAGS  ""
        # -pie Position independent execution is default on macOSX 10.6+
        CACHE STRING "Linker flags for security hardening")
    if(NOT DEVELOPER_BUILD)     # Strip debug symbols
        list(APPEND HARDENING_CFLAGS -O2)
        list(APPEND HARDENING_LDFLAGS -S)
    endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(HARDENING_CFLAGS
        -fstack-protector-strong    # Stack-based buffer overrun detection
        -fPIE -fPIC                 # Position independent
        -D_FORTIFY_SOURCE=2
        -Wformat -Wformat-security  # Format string vulnerability
        CACHE STRING "Compiler flags for security hardening")
    set(HARDENING_LDFLAGS
        -Wl,-z,noexecstack   # Stack execution protection
        -Wl,-z,relro,-z,now  # Data relocation protection
        -pie                 # Position independent
        CACHE STRING "Linker flags for security hardening")
    if(NOT DEVELOPER_BUILD)     # Strip debug symbols
        list(APPEND HARDENING_CFLAGS -O2)
        list(APPEND HARDENING_LDFLAGS -strip-debug)
    endif()
else()
    message(WARNING "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}. No security "
    "flags set")
endif()
include(CheckCXXCompilerFlag)
foreach(FLAG ${HARDENING_CFLAGS})
    check_cxx_compiler_flag(${FLAG} FLAG${FLAG})
    if (NOT FLAG${FLAG})
        list(REMOVE_ITEM HARDENING_CFLAGS ${FLAG})
        message(WARNING "Compiler does not support security option ${FLAG}")
    endif()
endforeach()
include(CheckLinkerFlag)
foreach(FLAG ${HARDENING_LDFLAGS})
    check_linker_flag(CXX ${FLAG} FLAG${FLAG})              # cmake 3.18+
    if (NOT FLAG${FLAG})
        list(REMOVE_ITEM HARDENING_LDFLAGS ${FLAG})
        message(WARNING "Linker does not support security option ${FLAG}")
    endif()
endforeach()
string(REPLACE ";" "," CUDA_HARDENING_CFLAGS "${HARDENING_CFLAGS}")
string(REPLACE ";" "," CUDA_HARDENING_LDFLAGS "${HARDENING_LDFLAGS}")
message(STATUS "Using security hardening compiler flags: ${HARDENING_CFLAGS} and
linker flags: ${HARDENING_LDFLAGS}")
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${CUDA_HARDENING_CFLAGS}>"
    "$<$<COMPILE_LANGUAGE:CXX>:${HARDENING_CFLAGS}>"
    )
add_link_options(
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${CUDA_HARDENING_LDFLAGS}>"
    "$<$<COMPILE_LANGUAGE:CXX>:${HARDENING_LDFLAGS}>"
    )

# Build CUDA module by default if CUDA is available
# Compatible with CMake 3.8+
if (BUILD_CUDA_MODULE)
    include(CheckLanguage)
    check_language(CUDA)
    message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    if(CMAKE_CUDA_COMPILER)
        find_package(CUDA REQUIRED) # required for cuda_select_nvcc_arch_flags
        message(STATUS "Building CUDA enabled")
        enable_language(CUDA)
        # TODO: when CMake minimum version is 3.18+, switch to the NEW behavior
        # of CMP0104 and replease CUDA_GENCODES with CUDA_ARCHITECTURES target
        # properties. See https://cmake.org/cmake/help/v3.18/policy/CMP0104.html.
        if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
            cmake_policy(SET CMP0104 OLD)
        endif()
        # Get gencode flags
        if("${CUDA_ARCH}" STREQUAL "User")
            cuda_select_nvcc_arch_flags(CUDA_GENCODES "${CUDA_ARCH_USER}")
        elseif("${CUDA_ARCH}" STREQUAL "BasicPTX")
            # Include oldest and most recent PTX and rely on JIT compilation
            set(CUDA_GENCODES "-gencode arch=compute_30,code=compute_30;-gencode arch=compute_75,code=compute_75")
        else()
            cuda_select_nvcc_arch_flags(CUDA_GENCODES "${CUDA_ARCH}")
        endif()
        # Make CUDA_GENCODES a string to avoid deduplication in target_compile_options
        string( REPLACE ";" " " CUDA_GENCODES "${CUDA_GENCODES}")
        message(STATUS "CUDA_GENCODES: ${CUDA_GENCODES}")
        # Allows CUDA headers to be included as "system" headers for CMake prior
        # to 3.17,
        # https://gitlab.kitware.com/cmake/cmake/-/issues/20270
        # https://github.com/intel-isl/CloudViewer/issues/2040
        if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
            list(REMOVE_ITEM CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        endif()
        message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
        message(STATUS "CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES}")
    else()
        set(BUILD_CUDA_MODULE OFF)
        message(STATUS "No CUDA support")
    endif()
endif ()

# OS specific settings
if(BUILD_GUI AND WIN32)
    if(NOT COMMAND target_link_options)  # CMake 3.13+ required
        # GLEW and CloudViewer make direct OpenGL calls and link to opengl32.lib;
        # Filament needs to link through bluegl.lib.
        # See https://github.com/google/filament/issues/652
        string(APPEND CMAKE_EXE_LINKER_FLAGS " /force:multiple")
        string(APPEND CMAKE_SHARED_LINKER_FLAGS " /force:multiple")
    endif()
endif()
if(WIN32)
    # Windows defaults to hidden symbol visibility, override that
    # TODO: It would be better to explictly export symbols.
    #       Then, we could use -fvisibility=hidden for Linux as well
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    add_definitions(-DGLEW_STATIC)		# enable GLEW for windows
    if(MSVC)
        # Make sure we don't hit the 65535 object member limit with MSVC
        #
        # /bigobj allows object files with more than 65535 members
        # /Ob2 enables function inlining, because MSVC is particularly
        # verbose with inline members
        #
        # See: https://github.com/tensorflow/tensorflow/pull/10962
        add_compile_options(/bigobj /Ob2)
        add_compile_options(/wd26812 /wd4819)
    endif()
    if (STATIC_WINDOWS_RUNTIME)
        # This needs cmake_policy(SET CMP0091 NEW)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

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

    foreach(dep IN LISTS CloudViewer_3RDPARTY_HEADER_TARGETS)
        if(TARGET ${dep})
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
            if(inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${inc})
            endif()
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
            if(inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${inc})
            endif()
            get_property(def TARGET ${dep} PROPERTY INTERFACE_COMPILE_DEFINITIONS)
            if(def)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ${def})
            endif()
        endif()
    endforeach()
endfunction()

# Check if the compiler defines the _GLIBCXX_USE_CXX11_ABI macro
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("#include <cxxabi.h>
int main() { return _GLIBCXX_USE_CXX11_ABI; }" HAS_GLIBCXX_USE_CXX11_ABI)

# Convenience function to set important target properties
function(cloudViewer_set_global_properties target)
    # Libraries need to be compiled with position independent code
    get_target_property(target_type ${target} TYPE)
    if (NOT target_type STREQUAL "EXECUTABLE")
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
    # Tell CMake we want a compiler that supports C++14 features
    target_compile_features(${target} PUBLIC
        cxx_variadic_templates
        cxx_constexpr
        cxx_override
        cxx_static_assert
        cxx_trailing_return_types
        cxx_return_type_deduction
    )
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp>
        $<INSTALL_INTERFACE:${CloudViewer_INSTALL_INCLUDE_DIR}>
    )
    if(BUILD_CUDA_MODULE)
        target_compile_definitions(${target} PRIVATE BUILD_CUDA_MODULE)
        if(BUILD_CACHED_CUDA_MANAGER)
            target_compile_definitions(${target} PRIVATE BUILD_CACHED_CUDA_MANAGER)
        endif()
    endif()
    if(USE_BLAS)
        target_compile_definitions(${target} PRIVATE USE_BLAS)
    endif()
    if(BUILD_RPC_INTERFACE)
         target_compile_definitions(${target} PRIVATE BUILD_RPC_INTERFACE ZMQ_STATIC)
    endif()
    if(GLIBCXX_USE_CXX11_ABI)
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
    else()
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
    endif()
    if(ENABLE_HEADLESS_RENDERING)
        target_compile_definitions(${target} PRIVATE HEADLESS_RENDERING)
    endif()
    if(NOT WITH_OPENMP)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>")
    endif()

    if(WIN32)
        target_compile_definitions(${target} PRIVATE
            WINDOWS
            _CRT_SECURE_NO_DEPRECATE
            _CRT_NONSTDC_NO_DEPRECATE
            _SCL_SECURE_NO_WARNINGS
        )
        if(MSVC)
            target_compile_definitions(${target} PRIVATE NOMINMAX _USE_MATH_DEFINES _ENABLE_EXTENDED_ALIGNED_STORAGE)
            target_compile_options(${target} PRIVATE /EHsc /wd4522 /wd4190 /wd4819 /Wv:18 /WX-)
            # Multi-thread compile, two ways to enable
            # Option 1, at build time: cmake --build . --parallel %NUMBER_OF_PROCESSORS%
            # https://stackoverflow.com/questions/36633074/set-the-number-of-threads-in-a-cmake-build
            # Option 2, at configure time: add /MP flag, no need to use Option 1
            # https://docs.microsoft.com/en-us/cpp/build/reference/mp-build-with-multiple-processes?view=vs-2019
            target_compile_options(${target} PRIVATE /MP)
            if(COMMAND target_link_options) # target_link_options add in 3.13
                if(BUILD_GUI)
                    # GLEW and CloudViewer make direct OpenGL calls and link to opengl32.lib;
                    # Filament needs to link through bluegl.lib.
                    # See https://github.com/google/filament/issues/652
                    target_link_options(${target} PRIVATE /force:multiple)
                endif()
                # The examples' .pdb files use up a lot of space and cause us to run
                # out of space on Github Actions. Compressing gives us another couple of GB.
                target_link_options(${target} PRIVATE /pdbcompress)
            endif()
        endif()
    elseif(APPLE)
        target_compile_definitions(${target} PRIVATE UNIX APPLE)
        target_compile_options(${target} PRIVATE "-Wno-deprecated-declarations")
    elseif(UNIX)
        target_compile_definitions(${target} PRIVATE UNIX)
        target_compile_options(${target} PRIVATE "-Wno-deprecated-declarations" "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-result>")
    endif()
    # The SHELL: prefix requires CMake 3.12+
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:${CUDA_GENCODES}>")
    # TBB static version is used
    # See: https://github.com/wjakob/tbb/commit/615d690c165d68088c32b6756c430261b309b79c
    target_compile_definitions(${target} PRIVATE __TBB_LIB_NAME=tbb_static)
    if(WITH_FAISS)
        target_compile_definitions(${target} PRIVATE WITH_FAISS)
    endif()
endfunction()

# recursively parse and return the entire directory tree.
# the result is placed in output
function(Directories root output)
    set(data "")
    list(APPEND data ${root})
    file(GLOB_RECURSE children LIST_DIRECTORIES true "${root}/*")
    list(SORT children)
    foreach(child ${children})
        if (IS_DIRECTORY ${child})
            list(APPEND data ${child})
        endif()
    endforeach()
    set (${output} ${data} PARENT_SCOPE)
endfunction()

function(cloudViewer_show_and_abort_on_warning trgt)
    if (BUILD_CUDA_MODULE)
        if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "10.2")
            set(CUDA_FLAGS "--compiler-options -Wall --Werror cross-execution-space-call,reorder,deprecated-declarations")
        else()
            set(CUDA_FLAGS "--Werror all-warnings,cross-execution-space-call,reorder,deprecated-declarations")
        endif()
    else()
        set(CUDA_FLAGS "")
    endif()

    set(DISABLE_MSVC_WARNINGS
            /wd4201        # non-standard extention nameless struct (filament includes)
            /wd4310        # cast truncates const value (filament)
            /wd4505        # unreferenced local function has been removed (dirent)
            /wd4127        # conditional expression is const (Eigen)
            /wd4189        # local variable is initialized but not referenced (PoissonRecon)
            /wd4324        # structure was padded due to alignment specifier (qhull)
            /wd4706        # assignment within conditional expression (fileIO, ...)
            /wd4100        # unreferenced parameter (many places in Open3D code)
            /wd4702        # unreachable code (many places in Open3D code)
            /wd4244        # implicit data type conversion (many places in Open3D code)
            /wd4245        # signed/unsigned mismatch (visualization, PoissonRecon, ...)
            )

    target_compile_options(${trgt} PRIVATE
        $<$<COMPILE_LANGUAGE:C>:$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX- ${DISABLE_MSVC_WARNINGS}>>
        $<$<COMPILE_LANGUAGE:C>:$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang,Intel>:-Wall>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX- ${DISABLE_MSVC_WARNINGS}>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang,Intel>:-Wall>>
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
        if(WIN32)
    	    foreach( filename "$<TARGET_FILE:${trgt}>" )
		        install_ext( FILES ${filename} "${CMAKE_INSTALL_PREFIX}/bin/${COLOUDVIEWER_APP_DIR_NAME}" "" )
            endforeach()
        endif()
    endif()
endfunction()

macro(ADD_SOURCE_GROUP MODULE_NAME)
    file(GLOB MODULE_HEADER_FILES "${MODULE_NAME}/*.h")
    source_group("Header Files\\${MODULE_NAME}" FILES ${MODULE_HEADER_FILES})
    file(GLOB MODULE_SOURCE_FILES "${MODULE_NAME}/*.cpp")
    source_group("Source Files\\${MODULE_NAME}" FILES ${MODULE_SOURCE_FILES})
endmacro(ADD_SOURCE_GROUP)

# ------------------------------------------------------------------------------
# OpenMP
# ------------------------------------------------------------------------------
if (WITH_OPENMP)
    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
        message(STATUS "Using installed OpenMP ${OpenMP_VERSION}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else ()
        message(STATUS "OpenMP NOT found")
    endif ()
 endif ()
 
 if (BUILD_RECONSTRUCTION)
    message(STATUS "Enabling Reconstruction support and automatically enable SIMD!")
    set(USE_SIMD ON)
endif()

# check for SSE flags
include( CMakeFindSSE )
if(USE_SIMD)
   CLOUDVIEWER_CHECK_FOR_SSE()
   set(CLOUDVIEWERCONFIG_SSE_DEFINITIONS "${SSE_DEFINITIONS}")
   set(CLOUDVIEWERCONFIG_SSE_COMPILE_OPTIONS ${SSE_FLAGS})
   if (MSVC)
       add_definitions(${CLOUDVIEWERCONFIG_SSE_DEFINITIONS})
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS_STR}")
   else()
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS_STR}")
   endif()

   message(STATUS "Enabling SIMD support")
   message(STATUS "Using CPU native flags for SSE optimization: ${CLOUDVIEWERCONFIG_SSE_DEFINITIONS}")
else()
   message(STATUS "Disabling SIMD support")
endif()

# Enumerate all third-party libraries which we need later
# This creates the necessary targets and sets the
# CloudViewer_3RDPARTY_*_TARGETS variables we use in cloudViewer_link_3rdparty_libraries
include(3rdparty/find_dependencies.cmake)

# Add external libraries
include( CMakeExternalLibs )

# Contrib. libraries (mainly for I/O)
include( AllSupport )

if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
    if( MSVC )
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin/RelWithDebInfo)
    else()
        SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endif()
endif()

add_subdirectory( core )

# Propagate the choice to use doubles for ScalarType
if ( ${CV_CORE_LIB_USES_DOUBLE} )
	add_definitions( -DSCALAR_TYPE_DOUBLE )
else()
	add_definitions( -DSCALAR_TYPE_FLOAT )
endif()

# Internal libs used by both ErowCloudViewer and cloudViewer python package
add_subdirectory( libs )
		  
# Examples
if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
    if( MSVC )
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/examples/Debug)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/examples/Release)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin/examples/RelWithDebInfo)
    else()
        SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endif()
endif()
add_subdirectory(examples)

# load plugins (potentially used by eCV)
if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
    if( MSVC )
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug/plugins)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release/plugins)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin/RelWithDebInfo/plugins)
    else()
        SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins)
    endif()
endif()
add_subdirectory( plugins )

# eCV
if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
    if( MSVC )
            SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
            SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)
            SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin/RelWithDebInfo)
    else()
        SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endif()
endif()
add_subdirectory( eCV )

# Install CMake configuration files
install(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}:: DESTINATION ${CloudViewer_INSTALL_CMAKE_DIR})
export(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}::)

# `make check-style` checks style for c++/cuda/python/ipynb files
add_custom_target(check-style
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/util/check_style.py
    COMMENT "Python executable used for style check: ${PYTHON_EXECUTABLE}."
)

# `make apply-style` applies style for c++/cuda/python/ipynb files
add_custom_target(apply-style
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/util/check_style.py --do_apply_style
    COMMENT "Python executable used for style check: ${PYTHON_EXECUTABLE}."
)

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
cloudviewer_aligned_print("Build Shared Library" "${BUILD_SHARED_LIBS}")
if(WIN32)
    cloudviewer_aligned_print("Use Windows Static Runtime" "${STATIC_WINDOWS_RUNTIME}")
endif()
cloudviewer_aligned_print("Build Unit Tests" "${BUILD_UNIT_TESTS}")
cloudviewer_aligned_print("Build Examples" "${BUILD_EXAMPLES}")
cloudviewer_aligned_print("Build Python Module" "${BUILD_PYTHON_MODULE}")
if(BUILD_PYTHON_MODULE)
    cloudviewer_aligned_print("- with Jupyter Notebook Support" "${BUILD_JUPYTER_EXTENSION}")
endif()
cloudviewer_aligned_print("Build Tensorflow Ops" "${BUILD_TENSORFLOW_OPS}")
cloudviewer_aligned_print("Build Pytorch Ops" "${BUILD_PYTORCH_OPS}")
cloudviewer_aligned_print("Build Benchmarks" "${BUILD_BENCHMARKS}")
cloudviewer_aligned_print("Bundle CloudViewer-ML" "${BUNDLE_CLOUDVIEWER_ML}")
cloudviewer_aligned_print("Build RPC interface" "${BUILD_RPC_INTERFACE}")

if(GLIBCXX_USE_CXX11_ABI)
    set(usage "1")
else()
    set(usage "0")
endif()
cloudviewer_aligned_print("Force GLIBCXX_USE_CXX11_ABI=" "${usage}")
message(STATUS "========================================================================")
message(STATUS "Third-Party Dependencies:")
set(deps
    EIGEN3
    FAISS
    FILAMENT
    FLANN
    FMT
    GLEW
    GLFW
    GOOGLETEST
    IMGUI
    IPPICV
    JPEG
    JSONCPP
    LIBLZF
    OPENGL
    PNG
    PYBIND11
    QHULL
    LIBREALSENSE
    TINYFILEDIALOGS
    TINYGLTF
    TINYOBJLOADER
)

foreach(dep IN ITEMS ${deps})
    if(${dep}_TARGET)
        if(NOT USE_SYSTEM_${dep})
            set(usage "yes (build from source)")
        else()
            set(usage "yes")
            if(${dep}_VERSION_STRING)
                set(usage "${usage} (v${${dep}_VERSION_STRING})")
            elseif(${dep}_VERSION)
                set(usage "${usage} (v${${dep}_VERSION})")
            endif()
        endif()
    else()
        set(usage "no")
    endif()
    cloudviewer_aligned_print("${dep}" "${usage}")
endforeach()
message(STATUS "========================================================================")
