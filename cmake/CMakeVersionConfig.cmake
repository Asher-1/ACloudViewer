# CMake Version Configuration
# 统一管理所有子项目的CMake最低版本要求

# 主项目的最低版本要求
set(CLOUDVIEWER_CMAKE_MINIMUM_VERSION "3.19" CACHE STRING "Minimum CMake version for main project")

# 子项目的最低版本要求
set(CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION "3.10" CACHE STRING "Minimum CMake version for subprojects")

# 插件的最低版本要求
set(CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION "3.10" CACHE STRING "Minimum CMake version for plugins")

# 第三方库的最低版本要求
set(CLOUDVIEWER_THIRDPARTY_CMAKE_MINIMUM_VERSION "3.10" CACHE STRING "Minimum CMake version for third-party libraries")

# 宏：为子项目设置cmake_minimum_required
macro(set_subproject_cmake_minimum_required)
    cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})
endmacro()

# 宏：为插件设置cmake_minimum_required
macro(set_plugin_cmake_minimum_required)
    cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})
endmacro()

# 宏：为第三方库设置cmake_minimum_required
macro(set_thirdparty_cmake_minimum_required)
    cmake_minimum_required(VERSION ${CLOUDVIEWER_THIRDPARTY_CMAKE_MINIMUM_VERSION})
endmacro()

# 预定义的版本宏，可以直接在文件开头使用
macro(CLOUDVIEWER_SUBPROJECT_MINIMUM_VERSION)
    cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})
endmacro()

macro(CLOUDVIEWER_PLUGIN_MINIMUM_VERSION)
    cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})
endmacro()

macro(CLOUDVIEWER_THIRDPARTY_MINIMUM_VERSION)
    cmake_minimum_required(VERSION ${CLOUDVIEWER_THIRDPARTY_CMAKE_MINIMUM_VERSION})
endmacro()

# 为了向后兼容，设置默认版本变量
if(NOT DEFINED CLOUDVIEWER_DEFAULT_CMAKE_MINIMUM_VERSION)
    set(CLOUDVIEWER_DEFAULT_CMAKE_MINIMUM_VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})
endif() 