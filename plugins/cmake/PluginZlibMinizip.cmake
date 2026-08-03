# cloudviewer_plugin_link_zlib_minizip
#
# Consistent bundled-zlib / system-minizip linking for plugins that need
# minizip (unzip.h) or direct libz symbols (e.g. assimp with hidden zlib).
#
# Priority:
#   1. 3rdparty_zlib  — vendored zlib + minizip (USE_SYSTEM_PNG=OFF)
#   2. 3rdparty_minizip — pkg-config minizip (WITH_MINIZIP=ON)
#   3. FATAL_ERROR when USE_SYSTEM_PNG=ON and neither target exists
#
# Arguments:
#   TARGET       — CMake target to link (required)
#   PLUGIN_NAME  — name shown in error messages (defaults to TARGET)
#   OUT_RAW_ZLIB — optional output variable for a direct libz link path
#                  (qSIBR: 3rdparty_zlib uses HIDDEN and assimp needs raw libz)
#   NO_LINK      — set raw libz path only; do not link 3rdparty_zlib/minizip on TARGET
#                  (qSIBR uses OUT_RAW_ZLIB in a separate link line)
function(cloudviewer_plugin_link_zlib_minizip target)
    cmake_parse_arguments(PZM "NO_LINK" "PLUGIN_NAME;OUT_RAW_ZLIB" "" ${ARGN})

    if(NOT target)
        message(FATAL_ERROR "cloudviewer_plugin_link_zlib_minizip: TARGET is required")
    endif()

    set(_plugin "${PZM_PLUGIN_NAME}")
    if(NOT _plugin)
        set(_plugin "${target}")
    endif()

    if(TARGET 3rdparty_zlib)
        if(NOT PZM_NO_LINK)
            target_link_libraries(${target} 3rdparty_zlib)
        endif()
        if(PZM_OUT_RAW_ZLIB)
            cloudviewer_plugin_extract_raw_zlib_path(_raw_zlib)
            set(${PZM_OUT_RAW_ZLIB} "${_raw_zlib}" PARENT_SCOPE)
        endif()
        return()
    endif()

    if(TARGET 3rdparty_minizip)
        if(NOT PZM_NO_LINK)
            target_link_libraries(${target} 3rdparty_minizip)
        endif()
        if(PZM_OUT_RAW_ZLIB)
            cloudviewer_plugin_extract_raw_zlib_from_minizip(_raw_zlib)
            set(${PZM_OUT_RAW_ZLIB} "${_raw_zlib}" PARENT_SCOPE)
        endif()
        return()
    endif()

    if(USE_SYSTEM_PNG)
        message(FATAL_ERROR
            "${_plugin} requires bundled zlib (USE_SYSTEM_PNG=OFF) "
            "or system minizip (WITH_MINIZIP=ON)")
    endif()
endfunction()

# Extract a direct libz path from 3rdparty_zlib (bypasses HIDDEN re-export).
function(cloudviewer_plugin_extract_raw_zlib_path out_var)
    set(_raw "")
    if(TARGET 3rdparty_zlib)
        get_target_property(_zlib_libs 3rdparty_zlib INTERFACE_LINK_LIBRARIES)
        if(_zlib_libs)
            foreach(_zl IN LISTS _zlib_libs)
                if(_zl MATCHES "libz\\." OR _zl MATCHES "zlib")
                    list(APPEND _raw "${_zl}")
                endif()
            endforeach()
        endif()
    endif()
    if(NOT _raw)
        find_package(ZLIB QUIET)
        if(ZLIB_FOUND)
            set(_raw ZLIB::ZLIB)
        endif()
    endif()
    set(${out_var} "${_raw}" PARENT_SCOPE)
endfunction()

# Prefer libz from minizip's pkg-config link line; fall back to minizip target.
function(cloudviewer_plugin_extract_raw_zlib_from_minizip out_var)
    set(_raw "")
    if(TARGET 3rdparty_minizip)
        get_target_property(_mz_libs 3rdparty_minizip INTERFACE_LINK_LIBRARIES)
        if(_mz_libs)
            foreach(_zl IN LISTS _mz_libs)
                if(_zl MATCHES "libz\\." OR _zl MATCHES "zlib" OR _zl STREQUAL "z")
                    list(APPEND _raw "${_zl}")
                endif()
            endforeach()
        endif()
    endif()
    if(NOT _raw)
        find_package(ZLIB QUIET)
        if(ZLIB_FOUND)
            set(_raw ZLIB::ZLIB)
        elseif(TARGET 3rdparty_minizip)
            set(_raw 3rdparty_minizip)
        endif()
    endif()
    set(${out_var} "${_raw}" PARENT_SCOPE)
endfunction()
