include(ExternalProject)

set(GLFW_LIB_NAME glfw3)

# Check for Wayland dependencies
# Note: GLFW 3.4 requires Wayland >= 1.20 (for WL_MARSHAL_FLAG_DESTROY)
# If system Wayland version is too old, Wayland support will be disabled
find_package(PkgConfig QUIET)

# Note: Only Ubuntu 22.04+ has Wayland 1.20+ packages (wayland-client wayland-protocols xkbcommon)
# WL_MARSHAL_FLAG_DESTROY was introduced in Wayland 1.20+
# GLFW 3.4 requires Wayland >= 1.20, so we always check the version
set(GLFW_BUILD_WAYLAND_OPTION OFF)
if(UNIX AND NOT APPLE)
    if(NOT PKG_CONFIG_FOUND)
        message(STATUS "pkg-config not found, disabling GLFW_BUILD_WAYLAND")
    elseif(NOT PKG_CONFIG_EXECUTABLE)
        message(STATUS "pkg-config executable not set, disabling GLFW_BUILD_WAYLAND")
    else()
        # Check if wayland packages are available via pkg-config
        execute_process(
            COMMAND ${PKG_CONFIG_EXECUTABLE} --exists wayland-client wayland-protocols xkbcommon
            RESULT_VARIABLE WAYLAND_PKG_RESULT
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(WAYLAND_PKG_RESULT EQUAL 0)
            # Check Wayland version - GLFW 3.4 requires >= 1.20
            execute_process(
                COMMAND ${PKG_CONFIG_EXECUTABLE} --modversion wayland-client
                OUTPUT_VARIABLE WAYLAND_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(WAYLAND_VERSION)
                # Compare version: need >= 1.20
                string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" VERSION_MATCH "${WAYLAND_VERSION}")
                if(VERSION_MATCH)
                    string(REGEX REPLACE "^([0-9]+)\\.([0-9]+).*" "\\1" WAYLAND_MAJOR "${WAYLAND_VERSION}")
                    string(REGEX REPLACE "^([0-9]+)\\.([0-9]+).*" "\\2" WAYLAND_MINOR "${WAYLAND_VERSION}")
                    if(WAYLAND_MAJOR GREATER 1 OR (WAYLAND_MAJOR EQUAL 1 AND WAYLAND_MINOR GREATER_EQUAL 20))
                        set(GLFW_BUILD_WAYLAND_OPTION ON)
                        message(STATUS "Wayland ${WAYLAND_VERSION} found (>= 1.20), enabling GLFW_BUILD_WAYLAND")
                    else()
                        message(STATUS "Wayland ${WAYLAND_VERSION} found but too old (need >= 1.20), disabling GLFW_BUILD_WAYLAND")
                    endif()
                else()
                    message(STATUS "Could not parse Wayland version '${WAYLAND_VERSION}', disabling GLFW_BUILD_WAYLAND")
                endif()
            else()
                message(STATUS "Could not determine Wayland version, disabling GLFW_BUILD_WAYLAND")
            endif()
        else()
            message(STATUS "Wayland dependencies not found via pkg-config, disabling GLFW_BUILD_WAYLAND")
        endif()
    endif()
endif()

ExternalProject_Add(
    ext_glfw
    PREFIX glfw
    URL https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz
    URL_HASH SHA256=c038d34200234d071fae9345bc455e4a8f2f544ab60150765d7704e08f3dac01
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/glfw"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS_hidden}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DGLFW_BUILD_EXAMPLES=OFF
        -DGLFW_BUILD_TESTS=OFF
        -DGLFW_BUILD_DOCS=OFF
        -DGLFW_BUILD_WAYLAND=${GLFW_BUILD_WAYLAND_OPTION}
        -DGLFW_BUILD_X11=ON
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${GLFW_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_glfw INSTALL_DIR)
set(GLFW_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(GLFW_LIB_DIR ${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR})
set(GLFW_LIBRARIES ${GLFW_LIB_NAME})
# Export Wayland option to parent scope so it can be used for compile definitions
# Note: On macOS, Wayland is always OFF, so this export is mainly for Linux
# The PARENT_SCOPE will only work if we're in a function or subdirectory
# If we're at top-level (like when included from find_dependencies.cmake),
# CMake will warn but it's harmless - the variable just won't be exported
# We suppress the warning on macOS since it's not needed there anyway
if(UNIX AND NOT APPLE)
    set(GLFW_BUILD_WAYLAND_OPTION ${GLFW_BUILD_WAYLAND_OPTION} PARENT_SCOPE)
endif()
