# References:
# https://cmake.org/cmake/help/v3.1/
# https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/

cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)

add_custom_target(build_all_3rd_party_libs
    COMMAND ${CMAKE_COMMAND} -E echo "Custom target build_all_3rd_party_libs reached."
)

set (ECV_CMAKE_SCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
option( CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING "change all dynamic dll output path to main project output path for debugging!" ON )

include( cmake/CMakePolicies.cmake )
include( cmake/CMakeSetCompilerOptions.cmake )

# Default debug suffix for libraries.
set( CMAKE_DEBUG_POSTFIX "d" )

# One shouldn't generate the BUILD project directly in the SOURCES folder!
if ( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )
        if ( NOT SAME_BUILD_AND_SOURCE_FOLDER_WARNING_ALREADY_ISSUED )
                message(FATAL_ERROR "It is not advised to BUILD the binaries directly in the SOURCE folder!\n If you want to proceed with this option, just CONFIGURE the project once again" )
                set( SAME_BUILD_AND_SOURCE_FOLDER_WARNING_ALREADY_ISSUED TRUE )
        endif()
endif()

# central location for specifying the CloudViewer version
file(STRINGS "version.txt" CLOUDVIEWER_VERSION_READ)
foreach(ver ${CLOUDVIEWER_VERSION_READ})
  if (ver MATCHES "CLOUDVIEWER_VERSION_(MAJOR|MINOR|PATCH|TWEAK) +([^ ]+)$")
    set(CLOUDVIEWER_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
  endif()
endforeach()
string(CONCAT CLOUDVIEWER_VERSION "${CLOUDVIEWER_VERSION_MAJOR}"
                             ".${CLOUDVIEWER_VERSION_MINOR}"
                             ".${CLOUDVIEWER_VERSION_PATCH}"
                             ".${CLOUDVIEWER_VERSION_TWEAK}")

project(ErowCloudViewerProjects VERSION ${CLOUDVIEWER_VERSION} LANGUAGES C CXX)
message(STATUS "ErowCloudViewerProjects ${PROJECT_VERSION}")

# npm version has to be MAJOR.MINOR.PATCH
string(CONCAT PROJECT_VERSION_THREE_NUMBER "${CLOUDVIEWER_VERSION_MAJOR}"
                                           ".${CLOUDVIEWER_VERSION_MINOR}"
                                           ".${CLOUDVIEWER_VERSION_PATCH}")

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
set(PROJECT_CODE    "https://github.com/intel-isl/Open3D")
set(PROJECT_ISSUES  "https://github.com/intel-isl/Open3D/issues")

# Default install folders
# (on Windows - msvc generator - the '_debug' suffix is automatically added for debug configurations)
set( INSTALL_DESTINATIONS ErowCloudViewer )

if(WIN32 AND NOT CYGWIN)
    set(DEF_INSTALL_CMAKE_DIR CMake)
else()
    set(DEF_INSTALL_CMAKE_DIR lib/cmake/CloudViewer)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
    "Installation directory for CMake files")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/3rdparty/CMake)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# config options
option(BUILD_SHARED_LIBS         "Build shared libraries"                   			OFF)
option(WITH_OPENMP               "Use OpenMP multi-threading"                           ON)
option(BUILD_CPP_EXAMPLES        "Build the CloudViewer example programs"        		ON)
option(BUILD_UNIT_TESTS          "Build the CloudViewer unit tests"                     OFF)
option(BUILD_EIGEN3              "Use the Eigen3 that comes with CloudViewer"    		ON)
option(BUILD_FLANN           	 "Use the flann that comes with CloudViewer"        	ON)
option(BUILD_FMT                 "Build fmtlib from source"                 			ON)
option(BUILD_GLEW                "Build glew from source"                   			OFF)
option(BUILD_GLFW                "Build glfw from source"                   			OFF)
option(BUILD_LIBLZF              "Build liblzf from source"                 			ON)
option(BUILD_PNG                 "Build png from source"                                OFF)
option(BUILD_JPEG                "Build jpeg-turbo from source"                         ON)
option(BUILD_PYBIND11            "Build pybind11 from source"                           ON)
option(BUILD_PYTHON_MODULE       "Build the python module"                              ON)
option(BUILD_LIBREALSENSE        "Build support for Intel RealSense camera"             OFF)
option(BUILD_AZURE_KINECT        "Build support for Azure Kinect sensor"                OFF)
option(BUILD_TINYGLTF            "Build TinyGLTF from source"               			ON)
option(BUILD_TINYOBJLOADER       "Build tinyobjloader from source"          			ON)
option(BUILD_TINYFILEDIALOGS     "Build tinyfiledialogs from source"                    ON)
option(BUILD_QHULL               "Build qhull from source"                  			ON)
option(ENABLE_JUPYTER            "Enable Jupyter support for CloudViewer"               ON)
option(STATIC_WINDOWS_RUNTIME    "Use static (MT/MTd) Windows runtime"                  OFF)
option(GLIBCXX_USE_CXX11_ABI     "Set -D_GLIBCXX_USE_CXX11_ABI=1"                       OFF)
# PCL ENGINE SUPPORT
option( USE_PCL_BACKEND "Check to use pcl library as backend" 							ON )

# In general, set the CXX ABI according to the other packages in your environment.
# See docs/compilation.rst for details.
if (GLIBCXX_USE_CXX11_ABI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

# headless rendering
if (ENABLE_HEADLESS_RENDERING)
    if (APPLE)
        message(FATAL_ERROR "Headless rendering is not supported on Mac OS")
        set(ENABLE_HEADLESS_RENDERING OFF)
    else()
        add_definitions(-DHEADLESS_RENDERING)
    endif()
endif()

# Set OpenGL policy
if (POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
    set(CMP0072_VALUE NEW)
    if (BUILD_GLFW)
        cmake_policy(SET CMP0072 OLD)
        set(CMP0072_VALUE OLD)
    endif()
endif()
# Set policy for hidden symbol visibility
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()
# Set policy to enable MSVC_RUNTIME_LIBRARY property
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()

# Set OS-specific things here
if (WIN32)
    # can't hide the unit testing option on Windows only
    # as a precaution: disable unit testing on Windows regardless of user input
    message(STATUS "Disable unit tests since this feature is not fully supported on Windows.")
    set(BUILD_UNIT_TESTS OFF)
    add_definitions(-DWINDOWS)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE -D_CRT_NONSTDC_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)		# suppress C4996 warning
    add_definitions(-DGLEW_STATIC)		# enable GLEW for windows
	add_compile_options(/wd4819)
	SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    message(STATUS "Compiling on Windows")
    if (MSVC)
        message(STATUS "Compiling with MSVC")
        add_definitions(-DNOMINMAX)
        add_definitions(-D_USE_MATH_DEFINES)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _ENABLE_EXTENDED_ALIGNED_STORAGE")

        # Ignore certain warnings
        # C4522: multiple assignment operators specified
        set(MSVC_WARNING_IGNORE "/wd4522")
		
        # Multi-thread compile, two ways to enable
        # Option 1, at build time: cmake --build . --parallel %NUMBER_OF_PROCESSORS%
        # https://stackoverflow.com/questions/36633074/set-the-number-of-threads-in-a-cmake-build
        # OPtion 2, at configure time: add /MP flag, no need to use Option 1
        # https://docs.microsoft.com/en-us/cpp/build/reference/mp-build-with-multiple-processes?view=vs-2019
        add_definitions("/MP")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MP ${MSVC_WARNING_IGNORE} /WX-")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MP ${MSVC_WARNING_IGNORE} /WX-")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MP ${MSVC_WARNING_IGNORE} /WX-")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MP ${MSVC_WARNING_IGNORE} /WX-")

        # /bigobj is enabled by default. On Windows, the build type (Debug, Release)
        # is set at build time, not config time, since CMAKE_BUILD_TYPE is not
        # supported on Windows.
        # ref: https://stackoverflow.com/a/15110915
        # ref: https://stackoverflow.com/a/19026241
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
    endif (MSVC)

    if (STATIC_WINDOWS_RUNTIME)
        # /MT  links libcmt.lib   libvcruntime.lib   libucrt.lib
        # /MTd links libcmtd.lib  libvcruntimed.lib  libucrtd.lib
        string(REPLACE "/MD" "/MT" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
        string(REPLACE "/MDd" "/MTd" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
        string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
        string(REPLACE "/MDd" "/MTd" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    else ()
        # By default, STATIC_WINDOWS_RUNTIME=OFF, also CMake defaults to
        # "/MD" or "/MDd"
        # /MD  links msvcrt.lib   vcruntime.lib      ucrt.lib
        # /MDd links msvcrtd.lib  vcruntimed.lib     ucrtd.lib
        string(REPLACE "/MT" "/MD" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
        string(REPLACE "/MTd" "/MDd" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
        string(REPLACE "/MT" "/MD" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
        string(REPLACE "/MTd" "/MDd" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    endif ()
    message(STATUS "CMAKE_C_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "CMAKE_C_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
    message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
elseif (CYGWIN)
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    message(STATUS "Compiling on Cygwin")
    add_definitions(-DCYGWIN)
elseif (APPLE)
    add_definitions(-DUNIX)
    add_compile_options(-Wno-deprecated-declarations)
    if (NOT BUILD_SHARED_LIBS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
    endif (NOT BUILD_SHARED_LIBS)
    # In Release build -O3 will be added automatically by CMake
    # We still enable -O3 at Debug build to optimize performance
    if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        add_definitions(-O3)
    endif()
elseif (UNIX)
    add_definitions(-DUNIX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    add_compile_options(-Wno-deprecated-declarations)
    add_compile_options(-Wno-unused-result)
    # In Release build -O3 will be added automatically by CMake
    # We still enable -O3 at debug build to optimize performance
    if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        add_definitions(-O3)
    endif()
    # disable BUILD_LIBREALSENSE since it is not fully supported on Linux
    message(STATUS "Compiling on Unix")
    message(STATUS "Disable RealSense since it is not fully supported on Linux.")
    set(BUILD_LIBREALSENSE OFF)
endif ()


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

function(ShowAndAbortOnWarning trgt)
  if(MSVC)
    # target_compile_options(${trgt} PRIVATE /W4 /WX)
    target_compile_options(${trgt} PRIVATE /W3)
  else()
    # target_compile_options(${trgt} PRIVATE -Wall -Wextra -Werror)
    target_compile_options(${trgt} PRIVATE -Wall -Werror)
  endif()
endfunction()

macro(ADD_SOURCE_GROUP MODULE_NAME)
    file(GLOB MODULE_HEADER_FILES "${MODULE_NAME}/*.h")
    source_group("Header Files\\${MODULE_NAME}" FILES ${MODULE_HEADER_FILES})
    file(GLOB MODULE_SOURCE_FILES "${MODULE_NAME}/*.cpp")
    source_group("Source Files\\${MODULE_NAME}" FILES ${MODULE_SOURCE_FILES})
endmacro(ADD_SOURCE_GROUP)


# 3rd-party projects that are added with external_project_add will be installed
# with this prefix. E.g.
# - 3RDPARTY_INSTALL_PREFIX: cloudViewer/build/3rdparty_install
# - Headers: cloudViewer/build/3rdparty_install/include/extern_lib.h
# - Libraries: cloudViewer/build/3rdparty_install/lib/extern_lib.a
set(3RDPARTY_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/3rdparty_install")

# 3rd-party libraries using the ExternalProject_Add approach will install
# headers in cloudViewer/build/3rdparty_install/include. We prioritize this include
# directory by putting it in front, to avoid mistakenly including other header
# files of the same name.
include_directories(${3RDPARTY_INSTALL_PREFIX}/include)

# 3rd-party libraries using the ExternalProject_Add approach will install
# libs in cloudViewer/build/3rdparty_install/lib.
# This isn't required for Ubuntu/Mac since the link directory info is propagated
# with the interface library. We still need this for Windows.
link_directories(${3RDPARTY_INSTALL_PREFIX}/lib)

# Handling dependencies
add_subdirectory(3rdparty)
link_directories(${3RDPARTY_LIBRARY_DIRS})

# RPATH Linux/Unix: (dynamic) libs are put in a subdir of prefix/lib,
# since they are only used by qCC/ccViewer
if( UNIX AND NOT APPLE )
	include( GNUInstallDirs )
	set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}/erowcloudviewer")
endif()

# Define target folders
# (now that ccViewer can have its own plugins, qCC and ccViewer must fall in separate folders!
if(WIN32) 
	set( EROWCLOUDVIEWER_DEST_FOLDER ErowCloudViewer )
	if (OPTION_GL_QUAD_BUFFER_SUPPORT)
		set ( EROWCLOUDVIEWER_DEST_FOLDER ${EROWCLOUDVIEWER_DEST_FOLDER}Stereo )
	endif()
else()
	set( EROWCLOUDVIEWER_DEST_FOLDER bin )
endif()

# set default install folders
if( WIN32 )
	# (on Windows - msvc generator - the '_debug' suffix is automatically added for debug configurations)
	set( INSTALL_DESTINATIONS ${EROWCLOUDVIEWER_DEST_FOLDER} )
elseif( UNIX )
	set( INSTALL_DESTINATIONS ${CMAKE_INSTALL_PREFIX}/bin )
endif()

# Load advanced scripts
include( cmake/CMakeInclude.cmake )

if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
	if( MSVC )
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)
	else()
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
	endif()
endif()


# Propagate the choice to use doubles for ScalarType
if ( ${USE_SCALAR_TYPE_DOUBLE} )
	add_definitions( -DSCALAR_TYPE_DOUBLE )
else()
	add_definitions( -DSCALAR_TYPE_FLOAT )
endif()

# Add external libraries
include( cmake/CMakeExternalLibs.cmake )

# Contrib. libraries (mainly for I/O)
include( contrib/AllSupport.cmake )

# Suppress 3rdparty header warnings with SYSTEM
include_directories(
    SYSTEM
    ${3RDPARTY_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/3rdparty/librealsense/include
)

add_subdirectory( core )

# Libs requested by eCV
add_subdirectory( libs/eCV_db )
add_subdirectory( libs/eCV_io ) # must always be included after eCV_db (dependency)
add_subdirectory( libs/qcustomplot )

if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
	if( MSVC )
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/examples/Debug)
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/examples/Release)
	else()
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
	endif()
endif()

add_subdirectory( libs/OpenGLEngine )

# Configure a header file to pass the version settings to the source code
configure_file("${PROJECT_SOURCE_DIR}/examples/Cpp/CloudViewer.h.in"
               "${PROJECT_SOURCE_DIR}/examples/Cpp/CloudViewer.h")
			  
# Examples
add_subdirectory(examples)

if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
	if( MSVC )
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)
	else()
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
	endif()
endif()

# PCL ENGINE LIB
if (USE_PCL_BACKEND) 
	include( cmake/CMakePCLLibs.cmake )
	add_subdirectory( libs/PCLEngine ) # must always be included after eCV_db and eCV_io (dependency)
endif()

# ECV PYTHON LIB
if (BUILD_PYTHON_MODULE)
    add_subdirectory( libs/Python )
endif()

if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
	if( MSVC )
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug/plugins)
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release/plugins)
	else()
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/plugins)
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/plugins)
	endif()
endif()

# load plugins (potentially used by eCV)
add_subdirectory( plugins )

if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
	if( MSVC )
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)
	else()
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
	endif()
endif()

# eCV
add_subdirectory( eCV )

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
cloudviewer_aligned_print("Headless Rendering" "${ENABLE_HEADLESS_RENDERING}")
cloudviewer_aligned_print("Azure Kinect Support" "${BUILD_AZURE_KINECT}")
cloudviewer_aligned_print("Build Shared Library" "${BUILD_SHARED_LIBS}")
cloudviewer_aligned_print("Build Unit Tests" "${BUILD_UNIT_TESTS}")
cloudviewer_aligned_print("Build Examples" "${BUILD_CPP_EXAMPLES}")
cloudviewer_aligned_print("Build Python Module" "${BUILD_PYTHON_MODULE}")
if(BUILD_PYTHON_MODULE)
    cloudviewer_aligned_print("- with Jupyter Notebook Support" "${ENABLE_JUPYTER}")
endif()

if(GLIBCXX_USE_CXX11_ABI)
    set(usage "1")
else()
    set(usage "0")
endif()
cloudviewer_aligned_print("Force GLIBCXX_USE_CXX11_ABI=" "${usage}")
message(STATUS "========================================================================")
message(STATUS "Third-Party Dependencies:")
foreach(dep IN ITEMS EIGEN3 FILAMENT FLANN FMT GLEW GLFW GOOGLETEST JPEG JSONCPP LIBLZF OPENGL PNG PYBIND11 QHULL LIBREALSENSE TINYFILEDIALOGS TINYGLTF TINYOBJLOADER)
	if(BUILD_${dep})
        set(usage "yes (build from source)")
    else()
		# set(usage "yes")
		# if(${dep}_VERSION_STRING)
			# set(usage "${usage} (v${${dep}_VERSION_STRING})")
		# elseif(${dep}_VERSION)
			# set(usage "${usage} (v${${dep}_VERSION})")
		# endif()
        set(usage "no")
    endif()
    cloudviewer_aligned_print("${dep}" "${usage}")
endforeach()
message(STATUS "========================================================================")
