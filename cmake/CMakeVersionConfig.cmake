# CMake Version Configuration
# Centralized minimum CMake version requirements for all subprojects

# Main project minimum version
set(CLOUDVIEWER_CMAKE_MINIMUM_VERSION "3.19" CACHE STRING "Minimum CMake version for main project")

# Subproject minimum version
set(CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION "3.10" CACHE STRING "Minimum CMake version for subprojects")

# Plugin minimum version
set(CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION "3.10" CACHE STRING "Minimum CMake version for plugins")

# Third-party library minimum version
set(CLOUDVIEWER_THIRDPARTY_CMAKE_MINIMUM_VERSION "3.10" CACHE STRING "Minimum CMake version for third-party libraries")

# Macro: set cmake_minimum_required for subprojects
macro(set_subproject_cmake_minimum_required)
    cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})
endmacro()

# Macro: set cmake_minimum_required for plugins
macro(set_plugin_cmake_minimum_required)
    cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})
endmacro()

# Macro: set cmake_minimum_required for third-party libraries
macro(set_thirdparty_cmake_minimum_required)
    cmake_minimum_required(VERSION ${CLOUDVIEWER_THIRDPARTY_CMAKE_MINIMUM_VERSION})
endmacro()

# Predefined version macros for use at the top of CMakeLists files
macro(CLOUDVIEWER_SUBPROJECT_MINIMUM_VERSION)
    cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})
endmacro()

macro(CLOUDVIEWER_PLUGIN_MINIMUM_VERSION)
    cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})
endmacro()

macro(CLOUDVIEWER_THIRDPARTY_MINIMUM_VERSION)
    cmake_minimum_required(VERSION ${CLOUDVIEWER_THIRDPARTY_CMAKE_MINIMUM_VERSION})
endmacro()

# Default version variable for backward compatibility
if(NOT DEFINED CLOUDVIEWER_DEFAULT_CMAKE_MINIMUM_VERSION)
    set(CLOUDVIEWER_DEFAULT_CMAKE_MINIMUM_VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})
endif()
