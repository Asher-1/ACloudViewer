#
# CloudViewer 3rd party library integration
#
set(CloudViewer_3RDPARTY_DIR "${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "CloudViewer_3RDPARTY_DIR :" ${CloudViewer_3RDPARTY_DIR})

# EXTERNAL_MODULES
# CMake modules we depend on in our public interface. These are modules we
# need to find_package() in our CMake config script, because we will use their
# targets.
set(CloudViewer_3RDPARTY_EXTERNAL_MODULES)

## PUBLIC_TARGETS
## CMake targets we link against in our public interface. They are
## either locally defined and installed, or imported from an external module
## (see above).
#set(CloudViewer_3RDPARTY_PUBLIC_TARGETS_FROM_CUSTOM)

## HEADER_TARGETS
## CMake targets we use in our public interface, but as a special case we do not
## need to link against the library. This simplifies dependencies where we merely
## expose declared data types from other libraries in our public headers, so it
## would be overkill to require all library users to link against that dependency.
#set(CloudViewer_3RDPARTY_HEADER_TARGETS)
#
## PRIVATE_TARGETS
## CMake targets for dependencies which are not exposed in the public API. This
## will probably include HEADER_TARGETS, but also anything else we use internally.
#set(CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM)

# XXX_FROM_CUSTOM vs. XXX_FROM_SYSTEM
# - "FROM_CUSTOM": downloaded or compiled
# - "FROM_SYSTEM": installed with a system package manager, deteced by CMake

# PUBLIC_TARGETS
# CMake targets we link against in our public interface. They are either locally
# defined and installed, or imported from an external module (see above).
set(CloudViewer_3RDPARTY_PUBLIC_TARGETS_FROM_CUSTOM)
set(CloudViewer_3RDPARTY_PUBLIC_TARGETS_FROM_SYSTEM)

# HEADER_TARGETS
# CMake targets we use in our public interface, but as a special case we only
# need to link privately against the library. This simplifies dependencies
# where we merely expose declared data types from other libraries in our
# public headers, so it would be overkill to require all library users to link
# against that dependency.
set(CloudViewer_3RDPARTY_HEADER_TARGETS_FROM_CUSTOM)
set(CloudViewer_3RDPARTY_HEADER_TARGETS_FROM_SYSTEM)

# PRIVATE_TARGETS
# CMake targets for dependencies which are not exposed in the public API. This
# will include anything else we use internally.
set(CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM)
set(CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM)

set(CUSTOM_TARGET_PREFIX "CloudViewer")

if (WIN32)
    # EXTERNAL INSTALL DIR
    set(CLOUDVIEWER_EXTERNAL_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/external")
    if (NOT EXISTS ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR})
        file(MAKE_DIRECTORY ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR})
    endif ()
    if (NOT EXISTS ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/include)
        file(MAKE_DIRECTORY ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/include)
    endif ()
    if (NOT EXISTS ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/lib)
        file(MAKE_DIRECTORY ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/lib)
    endif ()
else ()
    # EXTERNAL INSTALL DIR
    set(CLOUDVIEWER_EXTERNAL_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/external")
endif ()
list( APPEND CMAKE_PREFIX_PATH ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/lib )

find_package(PkgConfig QUIET)

# build_3rdparty_library(name ...)
#
# Builds a third-party library from source
#
# Valid options:
#    PUBLIC
#        the library belongs to the public interface and must be installed
#    HEADER
#        the library headers belong to the public interface, but the library
#        itself is linked privately
#    INCLUDE_ALL
#        install all files in the include directories. Default is *.h, *.hpp
#    VISIBLE
#        Symbols from this library will be visible for use outside CloudViewer.
#        Required, for example, if it may throw exceptions that need to be
#        caught in client code.
#    DIRECTORY <dir>
#        the library source directory <dir> is either a subdirectory of
#        3rdparty/ or an absolute directory.
#    INCLUDE_DIRS <dir> [<dir> ...]
#        include headers are in the subdirectories <dir>. Trailing slashes
#        have the same meaning as with install(DIRECTORY). <dir> must be
#        relative to the library source directory.
#        If your include is "#include <x.hpp>" and the path of the file is
#        "path/to/libx/x.hpp" then you need to pass "path/to/libx/"
#        with the trailing "/". If you have "#include <libx/x.hpp>" then you
#        need to pass "path/to/libx".
#    SOURCES <src> [<src> ...]
#        the library sources. Can be omitted for header-only libraries.
#        All sources must be relative to the library source directory.
#    LIBS <target> [<target> ...]
#        extra link dependencies
#    DEPENDS <target> [<target> ...]
#        targets on which <name> depends on and that must be built before.
#
function(build_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER;INCLUDE_ALL;VISIBLE" "DIRECTORY" "INCLUDE_DIRS;SOURCES;LIBS;DEPENDS" ${ARGN})
    if (arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: build_3rdparty_library(${name} ${ARGN})")
    endif ()
    get_filename_component(arg_DIRECTORY "${arg_DIRECTORY}" ABSOLUTE BASE_DIR "${CloudViewer_3RDPARTY_DIR}")
    if (arg_SOURCES)
        add_library(${name} STATIC)
        set_target_properties(${name} PROPERTIES OUTPUT_NAME "${PROJECT_NAME}_${name}")
        cloudViewer_set_global_properties(${name})
    else ()
        add_library(${name} INTERFACE)
    endif ()
    if (arg_INCLUDE_DIRS)
        set(include_dirs)
        foreach (incl IN LISTS arg_INCLUDE_DIRS)
            list(APPEND include_dirs "${arg_DIRECTORY}/${incl}")
        endforeach ()
    else ()
        set(include_dirs "${arg_DIRECTORY}/")
    endif ()
    if (arg_SOURCES)
        foreach (src IN LISTS arg_SOURCES)
            get_filename_component(abs_src "${src}" ABSOLUTE BASE_DIR "${arg_DIRECTORY}")
            # Mark as generated to skip CMake's file existence checks
            set_source_files_properties(${abs_src} PROPERTIES GENERATED TRUE)
            target_sources(${name} PRIVATE ${abs_src})
        endforeach ()
        foreach (incl IN LISTS include_dirs)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else ()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif ()
            target_include_directories(${name} SYSTEM PUBLIC $<BUILD_INTERFACE:${incl_path}>)
        endforeach ()
        # Do not export symbols from 3rd party libraries outside the CloudViewer DSO.
        if (NOT arg_PUBLIC AND NOT arg_HEADER AND NOT arg_VISIBLE)
            set_target_properties(${name} PROPERTIES
                    C_VISIBILITY_PRESET hidden
                    CXX_VISIBILITY_PRESET hidden
                    CUDA_VISIBILITY_PRESET hidden
                    VISIBILITY_INLINES_HIDDEN ON
                    )
        endif ()
        if (arg_LIBS)
            target_link_libraries(${name} PRIVATE ${arg_LIBS})
        endif ()
    else ()
        foreach (incl IN LISTS include_dirs)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else ()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif ()
            target_include_directories(${name} SYSTEM INTERFACE $<BUILD_INTERFACE:${incl_path}>)
        endforeach ()
    endif ()
    if (BUILD_SHARED_LIBS OR arg_PUBLIC)
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets
                RUNTIME DESTINATION ${CloudViewer_INSTALL_BIN_DIR}
                ARCHIVE DESTINATION ${CloudViewer_INSTALL_LIB_DIR}
                LIBRARY DESTINATION ${CloudViewer_INSTALL_LIB_DIR}
                )
    endif ()
    if (arg_PUBLIC OR arg_HEADER)
        foreach (incl IN LISTS include_dirs)
            if (arg_INCLUDE_ALL)
                install(DIRECTORY ${incl}
                        DESTINATION ${CloudViewer_INSTALL_INCLUDE_DIR}/cloudViewer/3rdparty
                        )
            else ()
                install(DIRECTORY ${incl}
                        DESTINATION ${CloudViewer_INSTALL_INCLUDE_DIR}/cloudViewer/3rdparty
                        FILES_MATCHING
                        PATTERN "*.h"
                        PATTERN "*.hpp"
                        )
            endif ()
            target_include_directories(${name} INTERFACE $<INSTALL_INTERFACE:${CloudViewer_INSTALL_INCLUDE_DIR}/cloudViewer/3rdparty>)
        endforeach ()
    endif ()
    if (arg_DEPENDS)
        add_dependencies(${name} ${arg_DEPENDS})
    endif ()
    add_library(${PROJECT_NAME}::${name} ALIAS ${name})
endfunction()

# CMake arguments for configuring ExternalProjects. Use the second _hidden
# version by default.
set(ExternalProject_CMAKE_ARGS
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
        -DCMAKE_CUDA_COMPILER_LAUNCHER=${CMAKE_CUDA_COMPILER_LAUNCHER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
        -DCMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}
        -DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}
        -DCMAKE_INSTALL_LIBDIR=${CloudViewer_INSTALL_LIB_DIR}
        # Always build 3rd party code in Release mode. Ignored by multi-config
        # generators (XCode, MSVC). MSVC needs matching config anyway.
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_POLICY_DEFAULT_CMP0091:STRING=NEW
        -DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        )

# Keep 3rd party symbols hidden from CloudViewer user code. Do not use if 3rd party
# libraries throw exceptions that escape CloudViewer.
set(ExternalProject_CMAKE_ARGS_hidden
        ${ExternalProject_CMAKE_ARGS}
        # Apply LANG_VISIBILITY_PRESET to static libraries and archives as well
        -DCMAKE_POLICY_DEFAULT_CMP0063:STRING=NEW
        -DCMAKE_CXX_VISIBILITY_PRESET=hidden
        -DCMAKE_CUDA_VISIBILITY_PRESET=hidden
        -DCMAKE_C_VISIBILITY_PRESET=hidden
        -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
        )

# pkg_config_3rdparty_library(name ...)
#
# Creates an interface library for a pkg-config dependency.
#
# The function will set ${name}_FOUND to TRUE or FALSE
# indicating whether or not the library could be found.
#
# Valid options:
#    PUBLIC
#        the library belongs to the public interface and must be installed
#    HEADER
#        the library headers belong to the public interface, but the library
#        itself is linked privately
#    SEARCH_ARGS
#        the arguments passed to pkg_search_module()
#
function(pkg_config_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER" "" "SEARCH_ARGS" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: pkg_config_3rdparty_library(${name} ${ARGN})")
    endif()
    if(PKGCONFIG_FOUND)
        pkg_search_module(pc_${name} ${arg_SEARCH_ARGS})
    endif()
    if(pc_${name}_FOUND)
        message(STATUS "Using installed third-party library ${name} ${${name_uc}_VERSION}: ${pc_${name}_INCLUDE_DIRS}")
        add_library(${name} INTERFACE)
        target_include_directories(${name} SYSTEM INTERFACE ${pc_${name}_INCLUDE_DIRS})
        target_link_libraries(${name} INTERFACE ${pc_${name}_LINK_LIBRARIES})
        foreach(flag IN LISTS pc_${name}_CFLAGS_OTHER)
            if(flag MATCHES "-D(.*)")
                target_compile_definitions(${name} INTERFACE ${CMAKE_MATCH_1})
            endif()
        endforeach()
        if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
            install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
        endif()
        set(${name}_FOUND TRUE PARENT_SCOPE)
        add_library(${PROJECT_NAME}::${name} ALIAS ${name})
    else()
        message(STATUS "Unable to find installed third-party library ${name}")
        set(${name}_FOUND FALSE PARENT_SCOPE)
    endif()
endfunction()

# find_package_3rdparty_library(name ...)
#
# Creates an interface library for a find_package dependency.
#
# The function will set ${name}_FOUND to TRUE or FALSE
# indicating whether or not the library could be found.
#
# Valid options:
#    PUBLIC
#        the library belongs to the public interface and must be installed
#    HEADER
#        the library headers belong to the public interface, but the library
#        itself is linked privately
#    REQUIRED
#        finding the package is required
#    QUIET
#        finding the package is quiet
#    PACKAGE <pkg>
#        the name of the queried package <pkg> forwarded to find_package()
#    PACKAGE_VERSION_VAR <pkg_version>
#        the variable <pkg_version> where to find the version of the queried package <pkg> find_package().
#        If not provided, PACKAGE_VERSION_VAR will default to <pkg>_VERSION.
#    TARGETS <target> [<target> ...]
#        the expected targets to be found in <pkg>
#    INCLUDE_DIRS
#        the expected include directory variable names to be found in <pkg>.
#        If <pkg> also defines targets, use them instead and pass them via TARGETS option.
#    LIBRARIES
#        the expected library variable names to be found in <pkg>.
#        If <pkg> also defines targets, use them instead and pass them via TARGETS option.
#    PATHS
#        Paths with hardcoded guesses. Same as in find_package.
#    DEPENDS
#        Adds targets that should be build before "name" as dependency.
#
function(find_package_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER;REQUIRED;QUIET"
        "PACKAGE;VERSION;PACKAGE_VERSION_VAR"
        "TARGETS;INCLUDE_DIRS;LIBRARIES;PATHS;DEPENDS" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: find_package_3rdparty_library(${name} ${ARGN})")
    endif()
    if(NOT arg_PACKAGE)
        message(FATAL_ERROR "find_package_3rdparty_library: Expected value for argument PACKAGE")
    endif()
    if(NOT arg_PACKAGE_VERSION_VAR)
        set(arg_PACKAGE_VERSION_VAR "${arg_PACKAGE}_VERSION")
    endif()
    set(find_package_args "")
    if(arg_VERSION)
        list(APPEND find_package_args "${arg_VERSION}")
    endif()
    if(arg_REQUIRED)
        list(APPEND find_package_args "REQUIRED")
    endif()
    if(arg_QUIET)
        list(APPEND find_package_args "QUIET")
    endif()
    if (arg_PATHS)
        list(APPEND find_package_args PATHS ${arg_PATHS} NO_DEFAULT_PATH)
    endif()
    find_package(${arg_PACKAGE} ${find_package_args})
    if(${arg_PACKAGE}_FOUND)
        message(STATUS "Using installed third-party library ${name} ${${arg_PACKAGE}_VERSION}")
        add_library(${name} INTERFACE)
        if(arg_TARGETS)
            foreach(target IN LISTS arg_TARGETS)
                if (TARGET ${target})
                    target_link_libraries(${name} INTERFACE ${target})
                else()
                    message(WARNING "Skipping undefined target ${target}")
                endif()
            endforeach()
        endif()
        if(arg_INCLUDE_DIRS)
            foreach(incl IN LISTS arg_INCLUDE_DIRS)
                target_include_directories(${name} INTERFACE ${${incl}})
            endforeach()
        endif()
        if(arg_LIBRARIES)
            foreach(lib IN LISTS arg_LIBRARIES)
                target_link_libraries(${name} INTERFACE ${${lib}})
            endforeach()
        endif()
        if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
            install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
            # Ensure that imported targets will be found again.
            if(arg_TARGETS)
                list(APPEND CloudViewer_3RDPARTY_EXTERNAL_MODULES ${arg_PACKAGE})
                set(CloudViewer_3RDPARTY_EXTERNAL_MODULES ${CloudViewer_3RDPARTY_EXTERNAL_MODULES} PARENT_SCOPE)
            endif()
        endif()
        if(arg_DEPENDS)
            add_dependencies(${name} ${arg_DEPENDS})
        endif()
        set(${name}_FOUND TRUE PARENT_SCOPE)
        set(${name}_VERSION ${${arg_PACKAGE_VERSION_VAR}} PARENT_SCOPE)
        add_library(${PROJECT_NAME}::${name} ALIAS ${name})
    else()
        message(STATUS "Unable to find installed third-party library ${name}")
        set(${name}_FOUND FALSE PARENT_SCOPE)
    endif()
endfunction()

# List of linker options for libCloudViewer client binaries (eg: pybind) to hide CloudViewer 3rd
# party dependencies. Only needed with GCC, not AppleClang.
set(CLOUDVIEWER_HIDDEN_3RDPARTY_LINK_OPTIONS)

if (CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
    find_library(LexLIB libl.a)    # test archive in macOS
    if (LexLIB)
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_LINK_OPTIONS -load_hidden ${LexLIB})
        check_cxx_source_compiles("int main() {return 0;}" FLAG_load_hidden)
        unset(CMAKE_REQUIRED_LINK_OPTIONS)
    endif ()
endif ()
if (NOT FLAG_load_hidden)
    set(FLAG_load_hidden 0)
endif ()

# import_3rdparty_library(name ...)
#
# Imports a third-party library that has been built independently in a sub project.
#
# Valid options:
#    PUBLIC
#        the library belongs to the public interface and must be installed
#    HEADER
#        the library headers belong to the public interface and will be
#        installed, but the library is linked privately.
#    INCLUDE_ALL
#        install all files in the include directories. Default is *.h, *.hpp
#    HIDDEN
#         Symbols from this library will not be exported to client code during
#         linking with CloudViewer. This is the opposite of the VISIBLE option in
#         build_3rdparty_library.  Prefer hiding symbols during building 3rd
#         party libraries, since this option is not supported by the MSVC linker.
#    GROUPED
#        add "-Wl,--start-group" libx.a liby.a libz.a "-Wl,--end-group" around
#        the libraries.
#    INCLUDE_DIRS
#        the temporary location where the library headers have been installed.
#        Trailing slashes have the same meaning as with install(DIRECTORY).
#        If your include is "#include <x.hpp>" and the path of the file is
#        "/path/to/libx/x.hpp" then you need to pass "/path/to/libx/"
#        with the trailing "/". If you have "#include <libx/x.hpp>" then you
#        need to pass "/path/to/libx".
#    LIBRARIES
#        the built library name(s). It is assumed that the library is static.
#        If the library is PUBLIC, it will be renamed to CloudViewer_${name} at
#        install time to prevent name collisions in the install space.
#    LIB_DIR
#        the temporary location of the library. Defaults to
#        CMAKE_ARCHIVE_OUTPUT_DIRECTORY.
#    DEPENDS <target> [<target> ...]
#        targets on which <name> depends on and that must be built before.
#
function(import_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER;INCLUDE_ALL;HIDDEN;GROUPED" "LIB_DIR" "INCLUDE_DIRS;LIBRARIES;DEPENDS" ${ARGN})
    if (arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: import_3rdparty_library(${name} ${ARGN})")
    endif ()
    if (NOT arg_LIB_DIR)
        set(arg_LIB_DIR "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
    endif ()
    add_library(${name} INTERFACE)
    if (arg_INCLUDE_DIRS)
        foreach (incl IN LISTS arg_INCLUDE_DIRS)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else ()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif ()
            target_include_directories(${name} SYSTEM INTERFACE $<BUILD_INTERFACE:${incl_path}>)
            if (arg_PUBLIC OR arg_HEADER)
                if (arg_INCLUDE_ALL)
                    install(DIRECTORY ${incl}
                            DESTINATION ${CloudViewer_INSTALL_INCLUDE_DIR}/cloudViewer/3rdparty
                            )
                else ()
                    install(DIRECTORY ${incl}
                            DESTINATION ${CloudViewer_INSTALL_INCLUDE_DIR}/cloudViewer/3rdparty
                            FILES_MATCHING
                            PATTERN "*.h"
                            PATTERN "*.hpp"
                            )
                endif ()
                target_include_directories(${name} INTERFACE $<INSTALL_INTERFACE:${CloudViewer_INSTALL_INCLUDE_DIR}/cloudViewer/3rdparty>)
            endif ()
        endforeach ()
    endif ()
    if (arg_LIBRARIES)
        list(LENGTH arg_LIBRARIES libcount)
        if (arg_HIDDEN AND NOT arg_PUBLIC AND NOT arg_HEADER)
            set(HIDDEN 1)
        else ()
            set(HIDDEN 0)
        endif ()
        if(arg_GROUPED AND UNIX AND NOT APPLE)
            target_link_libraries(${name} INTERFACE "-Wl,--start-group")
        endif()
        foreach (arg_LIBRARY IN LISTS arg_LIBRARIES)
            set(library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${arg_LIBRARY}${CMAKE_STATIC_LIBRARY_SUFFIX})
            if (libcount EQUAL 1)
                set(installed_library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${PROJECT_NAME}_${name}${CMAKE_STATIC_LIBRARY_SUFFIX})
            else ()
                set(installed_library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${PROJECT_NAME}_${name}_${arg_LIBRARY}${CMAKE_STATIC_LIBRARY_SUFFIX})
            endif ()

            # message("thirdparty lib: ${arg_LIB_DIR}/${library_filename}")
            # Apple compiler ld
            target_link_libraries(${name} INTERFACE
                    "$<BUILD_INTERFACE:$<$<AND:${HIDDEN},${FLAG_load_hidden}>:-load_hidden >${arg_LIB_DIR}/${library_filename}>")
            if (NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
                target_link_libraries(${name} INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CloudViewer_INSTALL_LIB_DIR}/${installed_library_filename}>)
            endif ()
            if (HIDDEN)
                # GNU compiler ld
                target_link_options(${name} INTERFACE
                        $<$<CXX_COMPILER_ID:GNU>:LINKER:--exclude-libs,${library_filename}>)
                list(APPEND CLOUDVIEWER_HIDDEN_3RDPARTY_LINK_OPTIONS $<$<CXX_COMPILER_ID:GNU>:LINKER:--exclude-libs,${library_filename}>)
                set(CLOUDVIEWER_HIDDEN_3RDPARTY_LINK_OPTIONS ${CLOUDVIEWER_HIDDEN_3RDPARTY_LINK_OPTIONS} PARENT_SCOPE)
            endif ()
        endforeach ()
        if(arg_GROUPED AND UNIX AND NOT APPLE)
            target_link_libraries(${name} INTERFACE "-Wl,--end-group")
        endif()
    endif ()
    if (NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
    endif ()
    if (arg_DEPENDS)
        add_dependencies(${name} ${arg_DEPENDS})
    endif ()
    add_library(${PROJECT_NAME}::${name} ALIAS ${name})
endfunction()

function(import_shared_3rdparty_library name ext_target)
    cmake_parse_arguments(arg "PUBLIC;HEADER" "LIB_DIR" "INCLUDE_DIRS;LIBRARIES" ${ARGN})
    if (arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: import_shared_3rdparty_library(${name} ${ARGN})")
    endif ()
    if (NOT arg_LIB_DIR)
        set(arg_LIB_DIR "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
    endif ()
    add_library(${name} INTERFACE)
    if (arg_INCLUDE_DIRS)
        foreach (incl IN LISTS arg_INCLUDE_DIRS)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else ()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif ()
            target_include_directories(${name} SYSTEM INTERFACE $<BUILD_INTERFACE:${incl_path}>)
            if (arg_PUBLIC OR arg_HEADER)
                install(DIRECTORY ${incl} DESTINATION ${CloudViewer_INSTALL_INCLUDE_DIR}/cloudViewer/3rdparty
                        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
                        )
                target_include_directories(${name} INTERFACE $<INSTALL_INTERFACE:${CloudViewer_INSTALL_INCLUDE_DIR}/cloudViewer/3rdparty>)
            endif ()
        endforeach ()
    endif ()
    if (arg_LIBRARIES)
        list(LENGTH arg_LIBRARIES libcount)
        foreach (arg_LIBRARY IN LISTS arg_LIBRARIES)
            set(library_filename ${CMAKE_SHARED_LIBRARY_PREFIX}${arg_LIBRARY}${CMAKE_SHARED_LIBRARY_SUFFIX})
            if (libcount EQUAL 1)
                set(installed_library_filename ${CMAKE_SHARED_LIBRARY_PREFIX}_${name}${CMAKE_SHARED_LIBRARY_SUFFIX})
            else ()
                set(installed_library_filename ${CMAKE_SHARED_LIBRARY_PREFIX}_${name}_${arg_LIBRARY}${CMAKE_SHARED_LIBRARY_SUFFIX})
            endif ()
            
            if (APPLE)
                # fix MacOS RPATH issues with shared ${ext_target}
                add_custom_command(TARGET ${ext_target}
                    POST_BUILD
                    COMMAND install_name_tool -id "@rpath/${library_filename}" ${arg_LIB_DIR}/${library_filename}
                    COMMAND ${CMAKE_COMMAND} -E
                    copy_if_different ${arg_LIB_DIR}/${library_filename} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/"
                )
                message(STATUS "install_name_tool -id "@rpath/${library_filename}" ${arg_LIB_DIR}/${library_filename}")
            else()
                # deploy for debugging
                add_custom_command(TARGET ${ext_target}
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E
                    copy_if_different ${arg_LIB_DIR}/${library_filename} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/"
                )
            endif()

            #message("shared thirdparty lib: ${arg_LIB_DIR}/${library_filename}")
            target_link_libraries(${name} INTERFACE $<BUILD_INTERFACE:${arg_LIB_DIR}/${library_filename}>)
            if (NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
                # install(FILES ${arg_LIB_DIR}/${library_filename}
                #         DESTINATION ${CloudViewer_INSTALL_LIB_DIR}
                #         RENAME ${installed_library_filename}
                #         )
                target_link_libraries(${name} INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CloudViewer_INSTALL_LIB_DIR}/${installed_library_filename}>)
            endif ()
        endforeach ()
    endif ()
    if (NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
    endif ()
    add_library(${PROJECT_NAME}::${name} ALIAS ${name})
endfunction()

function(copy_shared_library ext_target)
    cmake_parse_arguments(arg "PUBLIC;HEADER" "LIB_DIR" "LIBRARIES;SUFFIX" ${ARGN})

    if (arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: copy_shared_library(${name} ${ARGN})")
    endif ()
    if (NOT arg_LIB_DIR)
        set(arg_LIB_DIR "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
    endif ()

    if (NOT arg_SUFFIX)
        set(arg_SUFFIX "")
    endif ()

    if (arg_LIBRARIES)
        list(LENGTH arg_LIBRARIES libcount)
        foreach (arg_LIBRARY IN LISTS arg_LIBRARIES)
            set(library_filename ${CMAKE_SHARED_LIBRARY_PREFIX}${arg_LIBRARY}${CMAKE_SHARED_LIBRARY_SUFFIX})
            if (WIN32)
                add_custom_command(TARGET ${ext_target}
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E
                    copy_if_different ${arg_LIB_DIR}/${library_filename} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/${library_filename}"
                )
            elseif(arg_SUFFIX)
                set(library_filename ${library_filename}${arg_SUFFIX})
                    add_custom_command(TARGET ${ext_target}
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E
                    copy_if_different ${arg_LIB_DIR}/${library_filename} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${library_filename}"
                )
            endif()
        endforeach ()
    endif ()
endfunction()

#
# set_local_or_remote_url(url ...)
#
# If LOCAL_URL exists, set URL to LOCAL_URL, otherwise set URL to REMOTE_URLS.
# This function is needed since CMake does not allow specifying remote URL(s) if
# a local URL is specified.
#
# Valid options:
#    LOCAL_URL
#        local url to a file. Optional parameter. If the file does not exist,
#        LOCAL_URL will be ignored. If the file exists, REMOTE URLS will be
#        ignored. CMake only allows setting single LOCAL_URL for external
#        projects.
#    REMOTE_URLS
#        remote url(s) to download a file. CMake will try to download the file
#        in the specified order.
#
function(set_local_or_remote_url URL)
    cmake_parse_arguments(arg "" "LOCAL_URL" "REMOTE_URLS" ${ARGN})
    if (arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Invalid syntax: set_local_or_remote_url(${name} ${ARGN})")
    endif ()
    if (arg_LOCAL_URL AND (EXISTS ${arg_LOCAL_URL}))
        message(STATUS "Using local url: ${arg_LOCAL_URL}")
        set(${URL} "${arg_LOCAL_URL}" PARENT_SCOPE)
    else ()
        message(STATUS "Using remote url(s): ${arg_REMOTE_URLS}")
        set(${URL} "${arg_REMOTE_URLS}" PARENT_SCOPE)
    endif ()
endfunction()

include(ProcessorCount)
ProcessorCount(NPROC)

# CUDAToolkit
if (BUILD_CUDA_MODULE)
    find_package(CUDAToolkit REQUIRED)
endif ()

# Threads
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE) # -pthread instead of -lpthread
find_package_3rdparty_library(3rdparty_threads
        REQUIRED
        PACKAGE Threads
        TARGETS Threads::Threads
        )
list(APPEND CloudViewer_3RDPARTY_EXTERNAL_MODULES "Threads")

# Assimp
if(USE_SYSTEM_ASSIMP)
    find_package_3rdparty_library(3rdparty_assimp
        PACKAGE assimp
        TARGETS assimp::assimp
        DEPENDS ext_zlib
    )
    if(NOT 3rdparty_assimp_FOUND)
        set(USE_SYSTEM_ASSIMP OFF)
    endif()
endif()
if(NOT USE_SYSTEM_ASSIMP)
    include(${CloudViewer_3RDPARTY_DIR}/assimp/assimp.cmake)
    import_3rdparty_library(3rdparty_assimp
        INCLUDE_DIRS ${ASSIMP_INCLUDE_DIR}
        LIB_DIR      ${ASSIMP_LIB_DIR}
        LIBRARIES    ${ASSIMP_LIBRARIES}
        DEPENDS      ext_assimp
    )
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_assimp)
else()
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_assimp)
endif()

# OpenMP
if (WITH_OPENMP)
    find_package_3rdparty_library(3rdparty_openmp
            PACKAGE OpenMP
            PACKAGE_VERSION_VAR OpenMP_CXX_VERSION
            TARGETS OpenMP::OpenMP_CXX
            )
    if (3rdparty_openmp_FOUND)
        message(STATUS "Building with OpenMP")
        if (NOT BUILD_SHARED_LIBS)
            list(APPEND CloudViewer_3RDPARTY_EXTERNAL_MODULES "OpenMP")
        endif ()
        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_openmp)
    else()
        set(WITH_OPENMP OFF)
    endif ()
else ()
    set(WITH_OPENMP OFF)
endif ()


if (${GLIBCXX_USE_CXX11_ABI})
    set(CUSTOM_GLIBCXX_USE_CXX11_ABI 1)
else ()
    set(CUSTOM_GLIBCXX_USE_CXX11_ABI 0)
endif ()

# X11
if (UNIX AND NOT APPLE)
    find_package_3rdparty_library(3rdparty_x11
            QUIET
            PACKAGE X11
            TARGETS X11::X11
            )
    if (3rdparty_x11_FOUND)
        if (NOT BUILD_SHARED_LIBS)
            list(APPEND CloudViewer_3RDPARTY_EXTERNAL_MODULES "X11")
        endif ()
    endif ()
endif ()

# CUB (already included in CUDA 11.0+)
if (BUILD_CUDA_MODULE AND CUDAToolkit_VERSION VERSION_LESS "11.0")
    include(${CloudViewer_3RDPARTY_DIR}/cub/cub.cmake)
    import_3rdparty_library(3rdparty_cub
            INCLUDE_DIRS ${CUB_INCLUDE_DIRS}
            DEPENDS ext_cub
            )
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_cub)
endif ()

# cutlass
if (BUILD_CUDA_MODULE)
    include(${CloudViewer_3RDPARTY_DIR}/cutlass/cutlass.cmake)
    import_3rdparty_library(3rdparty_cutlass
            INCLUDE_DIRS ${CUTLASS_INCLUDE_DIRS}
            DEPENDS ext_cutlass
            )
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_cutlass)
endif ()

# Dirent
if (WIN32)
    build_3rdparty_library(3rdparty_dirent DIRECTORY dirent)
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_dirent)
endif ()

# Eigen3
if (USE_SYSTEM_EIGEN3)
    find_package_3rdparty_library(3rdparty_eigen3
            PUBLIC
            PACKAGE Eigen3
            TARGETS Eigen3::Eigen
            )
    if(NOT 3rdparty_eigen3_FOUND)
        set(USE_SYSTEM_EIGEN3 OFF)
    endif()
endif ()
if (NOT USE_SYSTEM_EIGEN3)
    include(${CloudViewer_3RDPARTY_DIR}/eigen/eigen.cmake)
    import_3rdparty_library(3rdparty_eigen3
            PUBLIC
            INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS}
            INCLUDE_ALL
            DEPENDS ext_eigen
            )
    list(APPEND CloudViewer_3RDPARTY_PUBLIC_TARGETS_FROM_CUSTOM 3rdparty_eigen3)
else()
    list(APPEND CloudViewer_3RDPARTY_PUBLIC_TARGETS_FROM_SYSTEM 3rdparty_eigen3)
endif()

# Nanoflann
if(USE_SYSTEM_NANOFLANN)
    find_package_3rdparty_library(3rdparty_nanoflann
        PACKAGE nanoflann
        VERSION 1.5.0
        TARGETS nanoflann::nanoflann
    )
    if(NOT 3rdparty_nanoflann_FOUND)
        set(USE_SYSTEM_NANOFLANN OFF)
    endif()
endif()
if(NOT USE_SYSTEM_NANOFLANN)
    include(${CloudViewer_3RDPARTY_DIR}/nanoflann/nanoflann.cmake)
    import_3rdparty_library(3rdparty_nanoflann
        INCLUDE_DIRS ${NANOFLANN_INCLUDE_DIRS}
        DEPENDS      ext_nanoflann
    )
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_nanoflann)
else()
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_nanoflann)
endif()

# GLEW
if (USE_SYSTEM_GLEW)
    find_package_3rdparty_library(3rdparty_glew
            HEADER
            PACKAGE GLEW
            TARGETS GLEW::GLEW
            )
    if (NOT 3rdparty_glew_FOUND)
        pkg_config_3rdparty_library(3rdparty_glew
            HEADER
            SEARCH_ARGS glew
        )
        if (NOT 3rdparty_glew_FOUND)
            set(USE_SYSTEM_GLEW OFF)
        endif ()
    endif ()
endif ()
if (NOT USE_SYSTEM_GLEW)
    build_3rdparty_library(3rdparty_glew DIRECTORY glew
            HEADER
            SOURCES
            src/glew.c
            INCLUDE_DIRS
            include/
            )
    if (ENABLE_HEADLESS_RENDERING)
        target_compile_definitions(3rdparty_glew PUBLIC GLEW_OSMESA)
    endif ()
    if (WIN32)
        target_compile_definitions(3rdparty_glew PUBLIC GLEW_STATIC)
    endif ()
    list(APPEND CloudViewer_3RDPARTY_HEADER_TARGETS_FROM_CUSTOM 3rdparty_glew)
else()
    list(APPEND CloudViewer_3RDPARTY_HEADER_TARGETS_FROM_SYSTEM 3rdparty_glew)
endif()

# GLFW
if (USE_SYSTEM_GLFW)
    find_package_3rdparty_library(3rdparty_glfw
            HEADER
            PACKAGE glfw3
            TARGETS glfw
            )
    if (NOT 3rdparty_glfw_FOUND)
        pkg_config_3rdparty_library(3rdparty_glfw
            HEADER
            SEARCH_ARGS glfw3
        )
        if (NOT 3rdparty_glfw_FOUND)
            set(USE_SYSTEM_GLFW OFF)
        endif ()
    endif ()
endif ()
if (NOT USE_SYSTEM_GLFW)
    message(STATUS "Building library 3rdparty_glfw from source")
    add_subdirectory(${CloudViewer_3RDPARTY_DIR}/GLFW)
    import_3rdparty_library(3rdparty_glfw
            HEADER
            INCLUDE_DIRS ${CloudViewer_3RDPARTY_DIR}/GLFW/include/
            LIBRARIES glfw3
            DEPENDS glfw
            )
    target_link_libraries(3rdparty_glfw INTERFACE 3rdparty_threads)
    if (UNIX AND NOT APPLE)
        find_library(RT_LIBRARY rt)
        if(RT_LIBRARY)
            target_link_libraries(3rdparty_glfw INTERFACE ${RT_LIBRARY})
        endif()
        find_library(MATH_LIBRARY m)
        if(MATH_LIBRARY)
            target_link_libraries(3rdparty_glfw INTERFACE ${MATH_LIBRARY})
        endif()
        if(CMAKE_DL_LIBS)
            target_link_libraries(3rdparty_glfw INTERFACE ${CMAKE_DL_LIBS})
        endif()
    endif ()
    if (APPLE)
        find_library(COCOA_FRAMEWORK Cocoa)
        find_library(IOKIT_FRAMEWORK IOKit)
        find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation)
        find_library(CORE_VIDEO_FRAMEWORK CoreVideo)
        target_link_libraries(3rdparty_glfw INTERFACE ${COCOA_FRAMEWORK} ${IOKIT_FRAMEWORK} ${CORE_FOUNDATION_FRAMEWORK} ${CORE_VIDEO_FRAMEWORK})
    endif ()
    if (WIN32)
        target_link_libraries(3rdparty_glfw INTERFACE gdi32)
    endif ()
    list(APPEND CloudViewer_3RDPARTY_HEADER_TARGETS_FROM_CUSTOM 3rdparty_glfw)
else()
    list(APPEND CloudViewer_3RDPARTY_HEADER_TARGETS_FROM_SYSTEM 3rdparty_glfw)
endif()
if(TARGET 3rdparty_x11)
    target_link_libraries(3rdparty_glfw INTERFACE 3rdparty_x11)
endif()

# TurboJPEG
if (USE_SYSTEM_JPEG AND BUILD_AZURE_KINECT)
    pkg_config_3rdparty_library(3rdparty_turbojpeg
        SEARCH_ARGS turbojpeg
    )
    if(3rdparty_turbojpeg_FOUND)
        message(STATUS "Using installed third-party library turbojpeg")
    else()
        message(STATUS "Unable to find installed third-party library turbojpeg")
        message(STATUS "Azure Kinect driver needs TurboJPEG API")
        set(USE_SYSTEM_JPEG OFF)
    endif()
endif ()

# JPEG
if (USE_SYSTEM_JPEG)
    find_package_3rdparty_library(3rdparty_jpeg
            PACKAGE JPEG
            TARGETS JPEG::JPEG
            )
    if (3rdparty_jpeg_FOUND)
        if (NOT BUILD_SHARED_LIBS)
            list(APPEND CloudViewer_3RDPARTY_EXTERNAL_MODULES "JPEG")
        endif ()
        if (TARGET 3rdparty_turbojpeg)
            list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_turbojpeg)
        endif ()
    else ()
        set(USE_SYSTEM_JPEG OFF)
    endif ()
endif ()
if (NOT USE_SYSTEM_JPEG)
    message(STATUS "Building third-party library JPEG from source")
    include(${CloudViewer_3RDPARTY_DIR}/libjpeg-turbo/libjpeg-turbo.cmake)
    import_3rdparty_library(3rdparty_jpeg
            INCLUDE_DIRS ${JPEG_TURBO_INCLUDE_DIRS}
            LIB_DIR ${JPEG_TURBO_LIB_DIR}
            LIBRARIES ${JPEG_TURBO_LIBRARIES}
            DEPENDS ext_turbojpeg
            )
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_jpeg)
else()
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_jpeg)
endif()

# jsoncpp: always compile from source to avoid ABI issues.
include(${CloudViewer_3RDPARTY_DIR}/jsoncpp/jsoncpp.cmake)
import_3rdparty_library(3rdparty_jsoncpp
        INCLUDE_DIRS ${JSONCPP_INCLUDE_DIRS}
        LIB_DIR ${JSONCPP_LIB_DIR}
        LIBRARIES ${JSONCPP_LIBRARIES}
        DEPENDS ext_jsoncpp
        )
list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_jsoncpp)

# draco: always compile from source to avoid ABI issues.
include(${CloudViewer_3RDPARTY_DIR}/draco/draco.cmake)
import_3rdparty_library(3rdparty_draco
        INCLUDE_DIRS ${DRACO_INCLUDE_DIRS}
        LIB_DIR ${DRACO_LIB_DIR}
        LIBRARIES ${DRACO_LIBRARIES}
        DEPENDS ext_draco
        )
set(DRACO_TARGET "3rdparty_draco")
list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_draco)

# liblzf
if (USE_SYSTEM_LIBLZF)
    find_package_3rdparty_library(3rdparty_liblzf
            PACKAGE liblzf
            TARGETS liblzf::liblzf
            )
    if(NOT 3rdparty_liblzf_FOUND)
        set(USE_SYSTEM_LIBLZF OFF)
    endif()
endif ()
if (NOT USE_SYSTEM_LIBLZF)
    build_3rdparty_library(3rdparty_liblzf DIRECTORY liblzf
            SOURCES
            liblzf/lzf_c.c
            liblzf/lzf_d.c
            )
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_liblzf)
else()
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_liblzf)
endif()

# tritriintersect
build_3rdparty_library(3rdparty_tritriintersect DIRECTORY tomasakeninemoeller
        INCLUDE_DIRS include/
        )
list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_tritriintersect)

# librealsense SDK
if (BUILD_LIBREALSENSE)
    if (USE_SYSTEM_LIBREALSENSE AND NOT GLIBCXX_USE_CXX11_ABI)
        # Turn off USE_SYSTEM_LIBREALSENSE.
        # Because it is affected by libraries built with different CXX ABIs.
        # See details: https://github.com/isl-org/CloudViewer/pull/2876
        message(STATUS "Set USE_SYSTEM_LIBREALSENSE=OFF, because GLIBCXX_USE_CXX11_ABI is OFF.")
        set(USE_SYSTEM_LIBREALSENSE OFF)
    endif ()
    if (USE_SYSTEM_LIBREALSENSE)
        find_package_3rdparty_library(3rdparty_librealsense
                PACKAGE realsense2
                TARGETS realsense2::realsense2
                )
        if(NOT 3rdparty_librealsense_FOUND)
            set(USE_SYSTEM_LIBREALSENSE OFF)
        endif()
    endif ()
    if (NOT USE_SYSTEM_LIBREALSENSE)
        include(${CloudViewer_3RDPARTY_DIR}/librealsense/librealsense.cmake)
        import_3rdparty_library(3rdparty_librealsense
                INCLUDE_DIRS ${LIBREALSENSE_INCLUDE_DIR}
                LIBRARIES ${LIBREALSENSE_LIBRARIES}
                LIB_DIR ${LIBREALSENSE_LIB_DIR}
                DEPENDS ext_librealsense
                )
        if (UNIX AND NOT APPLE)    # Ubuntu dependency: libudev-dev
            find_library(UDEV_LIBRARY udev REQUIRED
                    DOC "Library provided by the deb package libudev-dev")
            target_link_libraries(3rdparty_librealsense INTERFACE ${UDEV_LIBRARY})
        endif ()
        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_librealsense)
    else()
        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_librealsense)
    endif()
endif ()

# PNG
if (USE_SYSTEM_PNG)
    # ZLIB::ZLIB is automatically included by the PNG package.
    find_package_3rdparty_library(3rdparty_png
            PACKAGE PNG
            PACKAGE_VERSION_VAR PNG_VERSION_STRING
            TARGETS PNG::PNG
            )
    if(NOT 3rdparty_png_FOUND)
        set(USE_SYSTEM_PNG OFF)
    endif()
endif ()
if (NOT USE_SYSTEM_PNG)
    include(${CloudViewer_3RDPARTY_DIR}/zlib/zlib.cmake)
    import_3rdparty_library(3rdparty_zlib
            HIDDEN
            INCLUDE_DIRS  ${ZLIB_INCLUDE_DIRS}
            LIB_DIR       ${ZLIB_LIB_DIR}
            LIBRARIES     ${ZLIB_LIBRARIES}
            DEPENDS       ext_zlib
            )
    add_dependencies(ext_assimp ext_zlib)
    
    include(${CloudViewer_3RDPARTY_DIR}/libpng/libpng.cmake)
    import_3rdparty_library(3rdparty_png
            INCLUDE_DIRS ${LIBPNG_INCLUDE_DIRS}
            LIB_DIR ${LIBPNG_LIB_DIR}
            LIBRARIES ${LIBPNG_LIBRARIES}
            DEPENDS ext_libpng
            )
    add_dependencies(ext_libpng ext_zlib)
    target_link_libraries(3rdparty_png INTERFACE 3rdparty_zlib)
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_png)
else()
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_png)
endif()

# rply
build_3rdparty_library(3rdparty_rply DIRECTORY rply
        SOURCES
        rply/rply.c
        INCLUDE_DIRS
        rply/
        )
list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_rply)
list(APPEND CloudViewer_3RDPARTY_PUBLIC_TARGETS_FROM_CUSTOM 3rdparty_rply)

# tinyfiledialogs
build_3rdparty_library(3rdparty_tinyfiledialogs DIRECTORY tinyfiledialogs
        SOURCES
        include/tinyfiledialogs/tinyfiledialogs.c
        INCLUDE_DIRS
        include/
        )
list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_tinyfiledialogs)

# tinygltf
if (USE_SYSTEM_TINYGLTF)
    find_package_3rdparty_library(3rdparty_tinygltf
            PACKAGE TinyGLTF
            TARGETS TinyGLTF::TinyGLTF
            )
    if(NOT 3rdparty_tinygltf_FOUND)
        set(USE_SYSTEM_TINYGLTF OFF)
    endif()
endif ()
if (NOT USE_SYSTEM_TINYGLTF)
    include(${CloudViewer_3RDPARTY_DIR}/tinygltf/tinygltf.cmake)
    import_3rdparty_library(3rdparty_tinygltf
            INCLUDE_DIRS ${TINYGLTF_INCLUDE_DIRS}
            DEPENDS ext_tinygltf
            )
    target_compile_definitions(3rdparty_tinygltf INTERFACE TINYGLTF_IMPLEMENTATION STB_IMAGE_IMPLEMENTATION STB_IMAGE_WRITE_IMPLEMENTATION)
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_tinygltf)
else()
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_tinygltf)
endif()

# tinyobjloader
if (USE_SYSTEM_TINYOBJLOADER)
    find_package_3rdparty_library(3rdparty_tinyobjloader
            PACKAGE tinyobjloader
            TARGETS tinyobjloader::tinyobjloader
            )
    if(NOT 3rdparty_tinyobjloader_FOUND)
        set(USE_SYSTEM_TINYOBJLOADER OFF)
    endif()
endif ()
if (NOT USE_SYSTEM_TINYOBJLOADER)
    include(${CloudViewer_3RDPARTY_DIR}/tinyobjloader/tinyobjloader.cmake)
    import_3rdparty_library(3rdparty_tinyobjloader
            INCLUDE_DIRS ${TINYOBJLOADER_INCLUDE_DIRS}
            DEPENDS ext_tinyobjloader
            )
    target_compile_definitions(3rdparty_tinyobjloader INTERFACE TINYOBJLOADER_IMPLEMENTATION)
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_tinyobjloader)
else()
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_tinyobjloader)
endif()

# Qhullcpp
if(USE_SYSTEM_QHULLCPP)
    find_package_3rdparty_library(3rdparty_qhullcpp
        PACKAGE Qhull
        TARGETS Qhull::qhullcpp Qhull::qhull_r
    )
    if(NOT 3rdparty_qhullcpp_FOUND)
        set(USE_SYSTEM_QHULLCPP OFF)
    endif()
endif()
if (NOT USE_SYSTEM_QHULLCPP)
    include(${CloudViewer_3RDPARTY_DIR}/qhull/qhull.cmake)
    build_3rdparty_library(3rdparty_qhull_r DIRECTORY ${QHULL_SOURCE_DIR}
    SOURCES
        src/libqhull_r/global_r.c
        src/libqhull_r/stat_r.c
        src/libqhull_r/geom2_r.c
        src/libqhull_r/poly2_r.c
        src/libqhull_r/merge_r.c
        src/libqhull_r/libqhull_r.c
        src/libqhull_r/geom_r.c
        src/libqhull_r/poly_r.c
        src/libqhull_r/qset_r.c
        src/libqhull_r/mem_r.c
        src/libqhull_r/random_r.c
        src/libqhull_r/usermem_r.c
        src/libqhull_r/io_r.c
        src/libqhull_r/user_r.c
        src/libqhull_r/rboxlib_r.c
    INCLUDE_DIRS
        src/
    )
    build_3rdparty_library(3rdparty_qhullcpp DIRECTORY ${QHULL_SOURCE_DIR}
        SOURCES
            src/libqhullcpp/Coordinates.cpp
            src/libqhullcpp/PointCoordinates.cpp
            src/libqhullcpp/Qhull.cpp
            src/libqhullcpp/QhullFacet.cpp
            src/libqhullcpp/QhullFacetList.cpp
            src/libqhullcpp/QhullFacetSet.cpp
            src/libqhullcpp/QhullHyperplane.cpp
            src/libqhullcpp/QhullPoint.cpp
            src/libqhullcpp/QhullPointSet.cpp
            src/libqhullcpp/QhullPoints.cpp
            src/libqhullcpp/QhullQh.cpp
            src/libqhullcpp/QhullRidge.cpp
            src/libqhullcpp/QhullSet.cpp
            src/libqhullcpp/QhullStat.cpp
            src/libqhullcpp/QhullUser.cpp
            src/libqhullcpp/QhullVertex.cpp
            src/libqhullcpp/QhullVertexSet.cpp
            src/libqhullcpp/RboxPoints.cpp
            src/libqhullcpp/RoadError.cpp
            src/libqhullcpp/RoadLogEvent.cpp
        INCLUDE_DIRS
            src/
        )
    target_link_libraries(3rdparty_qhullcpp PRIVATE 3rdparty_qhull_r)
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_qhullcpp)
else()
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_qhullcpp)
endif()


# fmt
if(USE_SYSTEM_FMT)
    # MSVC >= 17.x required for building fmt 8+
    # SYCL / DPC++ needs fmt ver <8 or >= 9.2: https://github.com/fmtlib/fmt/issues/3005
    find_package_3rdparty_library(3rdparty_fmt
        PUBLIC
        PACKAGE fmt
        TARGETS fmt::fmt
    )
    if(NOT 3rdparty_fmt_FOUND)
        set(USE_SYSTEM_FMT OFF)
    endif()
endif()
if(NOT USE_SYSTEM_FMT)
    include(${CloudViewer_3RDPARTY_DIR}/fmt/fmt.cmake)
    import_3rdparty_library(3rdparty_fmt
        HEADER
        INCLUDE_DIRS ${FMT_INCLUDE_DIRS}
        LIB_DIR      ${FMT_LIB_DIR}
        LIBRARIES    ${FMT_LIBRARIES}
        DEPENDS      ext_fmt
    )
    # FMT 6.0, newer versions may require different flags
    target_compile_definitions(3rdparty_fmt INTERFACE FMT_HEADER_ONLY=0)
    target_compile_definitions(3rdparty_fmt INTERFACE FMT_USE_WINDOWS_H=0)
    target_compile_definitions(3rdparty_fmt INTERFACE FMT_STRING_ALIAS=1)
    list(APPEND CloudViewer_3RDPARTY_HEADER_TARGETS_FROM_CUSTOM 3rdparty_fmt)
else()
    list(APPEND CloudViewer_3RDPARTY_PUBLIC_TARGETS_FROM_SYSTEM 3rdparty_fmt)
endif()

# Pybind11
if (BUILD_PYTHON_MODULE)
    if (USE_SYSTEM_PYBIND11)
        find_package(pybind11)
    endif ()
    if (NOT USE_SYSTEM_PYBIND11 OR NOT TARGET pybind11::module)
        set(USE_SYSTEM_PYBIND11 OFF)
        include(${CloudViewer_3RDPARTY_DIR}/pybind11/pybind11.cmake)
        # pybind11 will automatically become available.
    endif ()
endif ()

# Azure Kinect
set(BUILD_AZURE_KINECT_COMMENT "//") # Set include header files in CloudViewer.h
if (BUILD_AZURE_KINECT)
    include(${CloudViewer_3RDPARTY_DIR}/azure_kinect/azure_kinect.cmake)
    import_3rdparty_library(3rdparty_k4a
            INCLUDE_DIRS ${K4A_INCLUDE_DIR}
            DEPENDS ext_k4a
            )
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_k4a)
endif ()

# PoissonRecon
include(${CloudViewer_3RDPARTY_DIR}/PoissonRecon/PoissonRecon.cmake)
import_3rdparty_library(3rdparty_poisson
        INCLUDE_DIRS ${POISSON_INCLUDE_DIRS}
        DEPENDS ext_poisson
        )
list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_poisson)

# Googletest
if (BUILD_UNIT_TESTS)
    if (USE_SYSTEM_GOOGLETEST)
        find_package_3rdparty_library(3rdparty_googletest
                PACKAGE GTest
                TARGETS GTest::gmock
                )
        if(NOT 3rdparty_googletest_FOUND)
            set(USE_SYSTEM_GOOGLETEST OFF)
        endif()
    endif ()
    if (NOT USE_SYSTEM_GOOGLETEST)
        include(${CloudViewer_3RDPARTY_DIR}/googletest/googletest.cmake)
        build_3rdparty_library(3rdparty_googletest DIRECTORY ${GOOGLETEST_SOURCE_DIR}
                SOURCES
                googletest/src/gtest-all.cc
                googlemock/src/gmock-all.cc
                INCLUDE_DIRS
                googletest/include/
                googletest/
                googlemock/include/
                googlemock/
                DEPENDS
                ext_googletest
                )
    endif ()
endif ()

# Google benchmark
if (BUILD_BENCHMARKS)
    include(${CloudViewer_3RDPARTY_DIR}/benchmark/benchmark.cmake)
    # benchmark and benchmark_main will automatically become available.
endif ()

# imgui
if (BUILD_GUI)
    if (USE_SYSTEM_IMGUI)
        find_package_3rdparty_library(3rdparty_imgui
                PACKAGE ImGui
                TARGETS ImGui::ImGui
                )
        if(NOT 3rdparty_imgui_FOUND)
            set(USE_SYSTEM_IMGUI OFF)
        endif()
    endif ()
    if (NOT USE_SYSTEM_IMGUI)
        include(${CloudViewer_3RDPARTY_DIR}/imgui/imgui.cmake)
        build_3rdparty_library(3rdparty_imgui DIRECTORY ${IMGUI_SOURCE_DIR}
                SOURCES
                imgui_demo.cpp
                imgui_draw.cpp
                imgui_widgets.cpp
                imgui.cpp
                DEPENDS
                ext_imgui
                )
        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_imgui)
    else()
        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_imgui)
    endif()
endif ()

# Filament
if(BUILD_GUI)
    if(USE_SYSTEM_FILAMENT)
        find_package_3rdparty_library(3rdparty_filament
            PACKAGE filament
            TARGETS filament::filament filament::geometry filament::image
        )
        if(3rdparty_filament_FOUND)
            set(FILAMENT_MATC "/usr/bin/matc")
        else()
            set(USE_SYSTEM_FILAMENT OFF)
        endif()
    endif()
    if(NOT USE_SYSTEM_FILAMENT)
        set(FILAMENT_RUNTIME_VER "")
        if(BUILD_FILAMENT_FROM_SOURCE)
            message(STATUS "Building third-party library Filament from source")
            if(MSVC OR (CMAKE_C_COMPILER_ID MATCHES ".*Clang" AND
                CMAKE_CXX_COMPILER_ID MATCHES ".*Clang"
                AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7))
                set(FILAMENT_C_COMPILER "${CMAKE_C_COMPILER}")
                set(FILAMENT_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
            else()
                message(STATUS "Filament can only be built with Clang >= 7")
                # First, check default version, because the user may have configured
                # a particular version as default for a reason.
                find_program(CLANG_DEFAULT_CC NAMES clang)
                find_program(CLANG_DEFAULT_CXX NAMES clang++)
                if(CLANG_DEFAULT_CC AND CLANG_DEFAULT_CXX)
                    execute_process(COMMAND ${CLANG_DEFAULT_CXX} --version OUTPUT_VARIABLE clang_version)
                    if(clang_version MATCHES "clang version ([0-9]+)")
                        if (CMAKE_MATCH_1 GREATER_EQUAL 7)
                            message(STATUS "Using ${CLANG_DEFAULT_CXX} to build Filament")
                            set(FILAMENT_C_COMPILER "${CLANG_DEFAULT_CC}")
                            set(FILAMENT_CXX_COMPILER "${CLANG_DEFAULT_CXX}")
                        endif()
                    endif()
                endif()
                # If the default version is not sufficient, look for some specific versions
                if(NOT FILAMENT_C_COMPILER OR NOT FILAMENT_CXX_COMPILER)
                    find_program(CLANG_VERSIONED_CC NAMES
                                 clang-19
                                 clang-18
                                 clang-17
                                 clang-16
                                 clang-15
                                 clang-14
                                 clang-13
                                 clang-12
                                 clang-11
                                 clang-10
                                 clang-9
                                 clang-8
                                 clang-7
                    )
                    find_program(CLANG_VERSIONED_CXX NAMES
                                 clang++-19
                                 clang++-18
                                 clang++-17
                                 clang++-16
                                 clang++-15
                                 clang++-14
                                 clang++-13
                                 clang++-12
                                 clang++-11
                                 clang++-10
                                 clang++-9
                                 clang++-8
                                 clang++-7
                    )
                    if (CLANG_VERSIONED_CC AND CLANG_VERSIONED_CXX)
                        set(FILAMENT_C_COMPILER "${CLANG_VERSIONED_CC}")
                        set(FILAMENT_CXX_COMPILER "${CLANG_VERSIONED_CXX}")
                        message(STATUS "Using ${CLANG_VERSIONED_CXX} to build Filament")
                    else()
                        message(FATAL_ERROR "Need Clang >= 7 to compile Filament from source")
                    endif()
                endif()
            endif()
            if (UNIX AND NOT APPLE)
                # Find corresponding libc++ and libc++abi libraries. On Ubuntu,
                # clang libraries are located at /usr/lib/llvm-{version}/lib,
                # and the default version will have a sybolic link at
                # /usr/lib/x86_64-linux-gnu/ or /usr/lib/aarch64-linux-gnu.
                #
                # On aarch64, the symbolic link path may not work for CMake's
                # find_library. Therefore, when compiling Filament from source,
                # we explicitly find the corresponding path based on the clang
                # version.
                execute_process(COMMAND ${FILAMENT_CXX_COMPILER} --version OUTPUT_VARIABLE clang_version)
                if(clang_version MATCHES "clang version ([0-9]+)")
                    set(CLANG_LIBDIR "/usr/lib/llvm-${CMAKE_MATCH_1}/lib")
                endif()
            endif()
            include(${CloudViewer_3RDPARTY_DIR}/filament/filament_build.cmake)
        else()
            message(STATUS "Using prebuilt third-party library Filament")
            include(${CloudViewer_3RDPARTY_DIR}/filament/filament_download.cmake)
            # Set lib directory for filament v1.9.9 on Windows.
            # Assume newer version if FILAMENT_PRECOMPILED_ROOT is set.
            if (WIN32 AND NOT FILAMENT_PRECOMPILED_ROOT)
                if (STATIC_WINDOWS_RUNTIME)
                    set(FILAMENT_RUNTIME_VER "x86_64/mt$<$<CONFIG:DEBUG>:d>")
                else()
                    set(FILAMENT_RUNTIME_VER "x86_64/md$<$<CONFIG:DEBUG>:d>")
                endif()
            endif()
        endif()
        if (APPLE)
            if (APPLE_AARCH64)
                set(FILAMENT_RUNTIME_VER arm64)
            else()
                set(FILAMENT_RUNTIME_VER x86_64)
            endif()
        endif()
        import_3rdparty_library(3rdparty_filament
            HEADER
            INCLUDE_DIRS ${FILAMENT_ROOT}/include/
            LIB_DIR ${FILAMENT_ROOT}/lib/${FILAMENT_RUNTIME_VER}
            LIBRARIES ${filament_LIBRARIES}
            DEPENDS ext_filament
        )
        set(FILAMENT_MATC "${FILAMENT_ROOT}/bin/matc")
        target_link_libraries(3rdparty_filament INTERFACE 3rdparty_threads ${CMAKE_DL_LIBS})
        if(UNIX AND NOT APPLE)
            # For ubuntu, llvm libs are located in /usr/lib/llvm-{version}/lib.
            # We first search for these paths, and then search CMake's default
            # search path. LLVM version must be >= 7 to compile Filament.
            if (NOT CLANG_LIBDIR)
                message(STATUS "Searching /usr/lib/llvm-[7..19]/lib/ for libc++ and libc++abi")
                foreach(llvm_ver RANGE 7 19)
                    set(llvm_lib_dir "/usr/lib/llvm-${llvm_ver}/lib")
                    find_library(CPP_LIBRARY    c++ PATHS ${llvm_lib_dir} NO_DEFAULT_PATH)
                    find_library(CPPABI_LIBRARY c++abi PATHS ${llvm_lib_dir} NO_DEFAULT_PATH)
                    if (CPP_LIBRARY AND CPPABI_LIBRARY)
                        set(CLANG_LIBDIR ${llvm_lib_dir})
                        message(STATUS "CLANG_LIBDIR found in ubuntu-default: ${CLANG_LIBDIR}")
                        set(LIBCPP_VERSION ${llvm_ver})
                        break()
                    endif()
                endforeach()
            endif()

            # Fallback to non-ubuntu-default paths. Note that the PATH_SUFFIXES
            # is not enforced by CMake.
            if (NOT CLANG_LIBDIR)
                message(STATUS "Clang C++ libraries not found. Searching other paths...")
                find_library(CPPABI_LIBRARY c++abi PATH_SUFFIXES
                             llvm-19/lib
                             llvm-18/lib
                             llvm-17/lib
                             llvm-16/lib
                             llvm-15/lib
                             llvm-14/lib
                             llvm-13/lib
                             llvm-12/lib
                             llvm-11/lib
                             llvm-10/lib
                             llvm-9/lib
                             llvm-8/lib
                             llvm-7/lib
                )
                file(REAL_PATH ${CPPABI_LIBRARY} CPPABI_LIBRARY)
                get_filename_component(CLANG_LIBDIR ${CPPABI_LIBRARY} DIRECTORY)
                string(REGEX MATCH "llvm-([0-9]+)/lib" _ ${CLANG_LIBDIR})
                set(LIBCPP_VERSION ${CMAKE_MATCH_1})
            endif()

            # Find clang libraries at the exact path ${CLANG_LIBDIR}.
            if (CLANG_LIBDIR)
                message(STATUS "Using CLANG_LIBDIR: ${CLANG_LIBDIR}")
            else()
                message(FATAL_ERROR "Cannot find matching libc++ and libc++abi libraries with version >=7.")
            endif()
            find_library(CPP_LIBRARY    c++    PATHS ${CLANG_LIBDIR} REQUIRED NO_DEFAULT_PATH)
            find_library(CPPABI_LIBRARY c++abi PATHS ${CLANG_LIBDIR} REQUIRED NO_DEFAULT_PATH)

            # Ensure that libstdc++ gets linked first.
            target_link_libraries(3rdparty_filament INTERFACE -lstdc++
                                  ${CPP_LIBRARY} ${CPPABI_LIBRARY})
            message(STATUS "Filament C++ libraries: ${CPP_LIBRARY} ${CPPABI_LIBRARY}")
            if (LIBCPP_VERSION GREATER 11)
                message(WARNING "libc++ (LLVM) version ${LIBCPP_VERSION} > 11 includes libunwind that " 
                "interferes with the system libunwind.so.8 and may crash Python code when exceptions "
                "are used. Please consider using libc++ (LLVM) v11.")
            endif()
        endif()
        if (APPLE)
            find_library(CORE_VIDEO CoreVideo)
            find_library(QUARTZ_CORE QuartzCore)
            find_library(OPENGL_LIBRARY OpenGL)
            find_library(METAL_LIBRARY Metal)
            find_library(APPKIT_LIBRARY AppKit)
            target_link_libraries(3rdparty_filament INTERFACE ${CORE_VIDEO} ${QUARTZ_CORE} ${OPENGL_LIBRARY} ${METAL_LIBRARY} ${APPKIT_LIBRARY})
            target_link_options(3rdparty_filament INTERFACE "-fobjc-link-runtime")
        endif()
        list(APPEND CloudViewer_3RDPARTY_HEADER_TARGETS_FROM_CUSTOM 3rdparty_filament)
    else()
        list(APPEND CloudViewer_3RDPARTY_HEADER_TARGETS_FROM_SYSTEM 3rdparty_filament)
    endif() # if(NOT USE_SYSTEM_FILAMENT)
endif()

# Headless rendering
if (ENABLE_HEADLESS_RENDERING)
    find_package_3rdparty_library(3rdparty_opengl
            REQUIRED
            PACKAGE OSMesa
            INCLUDE_DIRS OSMESA_INCLUDE_DIR
            LIBRARIES OSMESA_LIBRARY
            )
else ()
    find_package_3rdparty_library(3rdparty_opengl
            PACKAGE OpenGL
            TARGETS OpenGL::GL
            )
    set(USE_SYSTEM_OPENGL ON)
endif ()
list(APPEND CloudViewer_3RDPARTY_HEADER_TARGETS_FROM_SYSTEM 3rdparty_opengl)

# RPC interface
# zeromq
if(USE_SYSTEM_ZEROMQ)
    opkg_config_3rdparty_library(3rdparty_zeromq SEARCH_ARGS libzmq)
    if(NOT 3rdparty_zeromq_FOUND)
        set(USE_USE_SYSTEM_ZEROMQ OFF)
    endif()
endif()
if(NOT USE_SYSTEM_ZEROMQ)
    include(${CloudViewer_3RDPARTY_DIR}/zeromq/zeromq_build.cmake)
    import_3rdparty_library(3rdparty_zeromq
        HIDDEN
        INCLUDE_DIRS ${ZEROMQ_INCLUDE_DIRS}
        LIB_DIR      ${ZEROMQ_LIB_DIR}
        LIBRARIES    ${ZEROMQ_LIBRARIES}
        DEPENDS      ext_zeromq ext_cppzmq
    )
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_zeromq)
    if(DEFINED ZEROMQ_ADDITIONAL_LIBS)
        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM ${ZEROMQ_ADDITIONAL_LIBS})
    endif()
else()
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_zeromq)
    if(DEFINED ZEROMQ_ADDITIONAL_LIBS)
        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM ${ZEROMQ_ADDITIONAL_LIBS})
    endif()
endif()

# msgpack
include(${CloudViewer_3RDPARTY_DIR}/msgpack/msgpack_build.cmake)
import_3rdparty_library(3rdparty_msgpack
        INCLUDE_DIRS ${MSGPACK_INCLUDE_DIRS}
        DEPENDS ext_msgpack-c
        )
list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_msgpack)

# TBB
if(USE_SYSTEM_TBB)
    find_package_3rdparty_library(3rdparty_tbb
        PACKAGE TBB
        TARGETS TBB::tbb
    )
    if(NOT 3rdparty_tbb_FOUND)
        set(USE_SYSTEM_TBB OFF)
    endif()
endif()
if(NOT USE_SYSTEM_TBB)
    include(${CloudViewer_3RDPARTY_DIR}/mkl/tbb.cmake)
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_tbb)
else()
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_tbb)
endif()

# parallelstl
include(${CloudViewer_3RDPARTY_DIR}/parallelstl/parallelstl.cmake)
import_3rdparty_library(3rdparty_parallelstl
        PUBLIC
        INCLUDE_DIRS ${PARALLELSTL_INCLUDE_DIRS}
        INCLUDE_ALL
        DEPENDS ext_parallelstl
        )
list(APPEND CloudViewer_3RDPARTY_PUBLIC_TARGETS_FROM_SYSTEM 3rdparty_parallelstl)

# MKL/BLAS
if (USE_BLAS)
    if (USE_SYSTEM_BLAS)
        find_package(BLAS)
        find_package(LAPACK)
        find_package(LAPACKE)
        if(BLAS_FOUND AND LAPACK_FOUND AND LAPACKE_FOUND)
            message(STATUS "System BLAS/LAPACK/LAPACKE found.")
            list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM
                    ${BLAS_LIBRARIES}
                    ${LAPACK_LIBRARIES}
                    ${LAPACKE_LIBRARIES}
                    )
        else()
            message(STATUS "System BLAS/LAPACK/LAPACKE not found, setting USE_SYSTEM_BLAS=OFF.")
            set(USE_SYSTEM_BLAS OFF)
        endif()
    endif()

    if (NOT USE_SYSTEM_BLAS)
        # Install gfortran first for compiling OpenBLAS/Lapack from source.
        message(STATUS "Building OpenBLAS with LAPACK from source")
        set(BLAS_BUILD_FROM_SOURCE ON)

        find_program(gfortran_bin "gfortran")
        if (gfortran_bin)
            message(STATUS "gfortran found at ${gfortran}")
        else ()
            message(FATAL_ERROR "gfortran is required to compile LAPACK from source. "
                    "On Ubuntu, please install by `apt install gfortran`. "
                    "On macOS, please install by `brew install gfortran`. ")
        endif ()

        include(${CloudViewer_3RDPARTY_DIR}/openblas/openblas.cmake)
        import_3rdparty_library(3rdparty_blas
                HIDDEN
                INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIR}
                LIB_DIR ${OPENBLAS_LIB_DIR}
                LIBRARIES ${OPENBLAS_LIBRARIES}
                DEPENDS ext_openblas
                )

        # Get gfortran library search directories.
        execute_process(COMMAND ${gfortran_bin} -print-search-dirs
                OUTPUT_VARIABLE gfortran_search_dirs
                RESULT_VARIABLE RET
                OUTPUT_STRIP_TRAILING_WHITESPACE
                )
        if (RET AND NOT RET EQUAL 0)
            message(FATAL_ERROR "Failed to run `${gfortran_bin} -print-search-dirs`")
        endif ()

        # Parse gfortran library search directories into CMake list.
        string(REGEX MATCH "libraries: =(.*)" match_result ${gfortran_search_dirs})
        if (match_result)
            string(REPLACE ":" ";" gfortran_lib_dirs ${CMAKE_MATCH_1})
        else ()
            message(FATAL_ERROR "Failed to parse gfortran_search_dirs: ${gfortran_search_dirs}")
        endif ()

        if (LINUX_AARCH64 OR APPLE_AARCH64)
            # Find libgfortran.a and libgcc.a inside the gfortran library search
            # directories. This ensures that the library matches the compiler.
            # On ARM64 Ubuntu and ARM64 macOS, libgfortran.a is compiled with `-fPIC`.
            find_library(gfortran_lib NAMES libgfortran.a PATHS ${gfortran_lib_dirs} REQUIRED)
            find_library(gcc_lib NAMES libgcc.a PATHS ${gfortran_lib_dirs} REQUIRED)
            target_link_libraries(3rdparty_blas INTERFACE
                    ${gfortran_lib}
                    ${gcc_lib}
                    )
            if (APPLE_AARCH64)
                find_library(quadmath_lib NAMES libquadmath.a PATHS ${gfortran_lib_dirs} REQUIRED)
                target_link_libraries(3rdparty_blas INTERFACE
                        ${quadmath_lib})
                # Suppress Apple compiler warnigns.
                target_link_options(3rdparty_blas INTERFACE "-Wl,-no_compact_unwind")
            endif ()
        elseif (UNIX AND NOT APPLE)
            # On Ubuntu 20.04 x86-64, libgfortran.a is not compiled with `-fPIC`.
            # The temporary solution is to link the shared library libgfortran.so.
            # If we distribute a Python wheel, the user's system will also need
            # to have libgfortran.so preinstalled.
            #
            # If you have to link libgfortran.a statically
            # - Read https://gcc.gnu.org/wiki/InstallingGCC
            # - Run `gfortran --version`, e.g. you get 9.3.0
            # - Checkout gcc source code to the corresponding version
            # - Configure with
            #   ${PWD}/../gcc/configure --prefix=${HOME}/gcc-9.3.0 \
            #                           --enable-languages=c,c++,fortran \
            #                           --with-pic --disable-multilib
            # - make install -j$(nproc) # This will take a while
            # - Change this cmake file to libgfortran.a statically.
            # - Link
            #   - libgfortran.a
            #   - libgcc.a
            #   - libquadmath.a
            target_link_libraries(3rdparty_blas INTERFACE gfortran)
            target_link_libraries(3rdparty_blas INTERFACE Threads::Threads gfortran)
        endif ()
        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_blas)
    endif ()
else ()
    include(${CloudViewer_3RDPARTY_DIR}/mkl/mkl.cmake)
    # MKL, cuSOLVER, cuBLAS
    # We link MKL statically. For MKL link flags, refer to:
    # https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html
    message(STATUS "Using MKL to support BLAS and LAPACK functionalities.")
    import_3rdparty_library(3rdparty_blas
        GROUPED
        HIDDEN
        INCLUDE_DIRS ${STATIC_MKL_INCLUDE_DIR}
        LIB_DIR      ${STATIC_MKL_LIB_DIR}
        LIBRARIES    ${STATIC_MKL_LIBRARIES}
        DEPENDS      3rdparty_tbb ext_mkl_include ext_mkl
    )
    if(UNIX)
        target_compile_options(3rdparty_blas INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-m64>")
        target_link_libraries(3rdparty_blas INTERFACE 3rdparty_threads ${CMAKE_DL_LIBS})
    endif()
    target_compile_definitions(3rdparty_blas INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:MKL_ILP64>")
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_blas)
endif ()

# cuBLAS
if (BUILD_CUDA_MODULE)
    if (WIN32)
        # Nvidia does not provide static libraries for Windows. We don't release
        # pip wheels for Windows with CUDA support at the moment. For the pip
        # wheels to support CUDA on Windows out-of-the-box, we need to either
        # ship the CUDA toolkit with the wheel (e.g. PyTorch can make use of the
        # cudatoolkit conda package), or have a mechanism to locate the CUDA
        # toolkit from the system.
        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM CUDA::cudart CUDA::cusolver CUDA::cublas)
    else ()
        # CMake docs   : https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
        # cusolver 11.0: https://docs.nvidia.com/cuda/archive/11.0/cusolver/index.html#static-link-lapack
        # cublas   11.0: https://docs.nvidia.com/cuda/archive/11.0/cublas/index.html#static-library
        # The link order below is important. Theoretically we should use
        # find_package_3rdparty_library, but we have to insert
        # liblapack_static.a in the middle of the targets.
        add_library(3rdparty_cublas INTERFACE)
        if(CUDAToolkit_VERSION VERSION_LESS "12.0")
            target_link_libraries(3rdparty_cublas INTERFACE
                    CUDA::cusolver_static
                    ${CUDAToolkit_LIBRARY_DIR}/liblapack_static.a
                    CUDA::cusparse_static
                    CUDA::cublas_static
                    CUDA::cudart_static  # must link this to avoid hidden symbol `cudaMemcpy' in libcudart_static.a is referenced by DSO
                    CUDA::cublasLt_static
                    CUDA::culibos
                    )
        else()
            # In CUDA 12.0 the liblapack_static.a is deprecated and removed.
            # Use the libcusolver_lapack_static.a instead.
            # Use of static libraries is preferred.
            if(BUILD_WITH_CUDA_STATIC)
                # Use static CUDA libraries.
                target_link_libraries(3rdparty_cublas INTERFACE
                    CUDA::cusolver_static
                    ${CUDAToolkit_LIBRARY_DIR}/libcusolver_lapack_static.a
                    CUDA::cusparse_static
                    CUDA::cublas_static
                    CUDA::cublasLt_static
                    CUDA::culibos
                    CUDA::cudart_static
                )
            else()
                # Use shared CUDA libraries.
                target_link_libraries(3rdparty_cublas INTERFACE
                    CUDA::cusolver
                    ${CUDAToolkit_LIBRARY_DIR}/libcusolver.so
                    CUDA::cusparse
                    CUDA::cublas
                    CUDA::cublasLt
                    CUDA::culibos
                )
            endif()
        endif()
        if (NOT BUILD_SHARED_LIBS)
            # Listed in ${CMAKE_INSTALL_PREFIX}/lib/cmake/CloudViewer/${PROJECT_NAME}Targets.cmake.
            install(TARGETS 3rdparty_cublas EXPORT ${PROJECT_NAME}Targets)
            list(APPEND CloudViewer_3RDPARTY_EXTERNAL_MODULES "CUDAToolkit")
        endif ()
        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_cublas)
    endif ()
endif ()


# NPP
if (BUILD_CUDA_MODULE)
    # NPP library list: https://docs.nvidia.com/cuda/npp/index.html
    if (WIN32)
        find_package_3rdparty_library(3rdparty_cuda_npp
                REQUIRED
                PACKAGE CUDAToolkit
                TARGETS CUDA::nppc
                        CUDA::nppicc
                        CUDA::nppif
                        CUDA::nppig
                        CUDA::nppim
                        CUDA::nppial
        )
    else ()
        if(BUILD_WITH_CUDA_STATIC)
            # Use static CUDA libraries.
            find_package_3rdparty_library(3rdparty_cuda_npp
                REQUIRED
                PACKAGE CUDAToolkit
                TARGETS CUDA::nppc_static
                        CUDA::nppicc_static
                        CUDA::nppif_static
                        CUDA::nppig_static
                        CUDA::nppim_static
                        CUDA::nppial_static
            )
        else()
            # Use shared CUDA libraries.
            find_package_3rdparty_library(3rdparty_cuda_npp
                REQUIRED
                PACKAGE CUDAToolkit
                TARGETS CUDA::nppc
                        CUDA::nppicc
                        CUDA::nppif
                        CUDA::nppig
                        CUDA::nppim
                        CUDA::nppial
            )
        endif()
    endif ()

    if(NOT 3rdparty_cuda_npp_FOUND)
        message(FATAL_ERROR "CUDA NPP libraries not found.")
    endif()
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_cuda_npp)
endif ()

# IPP
if (WITH_IPPICV)
    # Ref: https://stackoverflow.com/a/45125525
    set(IPPICV_SUPPORTED_HW AMD64 x86_64 x64 x86 X86 i386 i686)
    # Unsupported: ARM64 aarch64 armv7l armv8b armv8l ...
    if (NOT CMAKE_HOST_SYSTEM_PROCESSOR IN_LIST IPPICV_SUPPORTED_HW)
        set(WITH_IPPICV OFF)
        message(WARNING "IPP-ICV disabled: Unsupported Platform.")
    else ()
        include(${CloudViewer_3RDPARTY_DIR}/ippicv/ippicv.cmake)
        if (WITH_IPPICV)
            message(STATUS "IPP-ICV ${IPPICV_VERSION_STRING} available. Building interface wrappers IPP-IW.")
            import_3rdparty_library(3rdparty_ippicv
                    HIDDEN
                    INCLUDE_DIRS ${IPPICV_INCLUDE_DIR}
                    LIBRARIES ${IPPICV_LIBRARIES}
                    LIB_DIR ${IPPICV_LIB_DIR}
                    DEPENDS ext_ippicv
                    )
            target_compile_definitions(3rdparty_ippicv INTERFACE ${IPPICV_DEFINITIONS})
            list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM 3rdparty_ippicv)
        endif ()
    endif ()
endif ()

# Stdgpu
if (BUILD_CUDA_MODULE)
    if(USE_SYSTEM_STDGPU)
        find_package_3rdparty_library(3rdparty_stdgpu
            PACKAGE stdgpu
            TARGETS stdgpu::stdgpu
        )
        if(NOT 3rdparty_stdgpu_FOUND)
            set(USE_SYSTEM_STDGPU OFF)
        endif()
    endif()
    if(NOT USE_SYSTEM_STDGPU)
        include(${CloudViewer_3RDPARTY_DIR}/stdgpu/stdgpu.cmake)
        import_3rdparty_library(3rdparty_stdgpu
            INCLUDE_DIRS ${STDGPU_INCLUDE_DIRS}
            LIB_DIR      ${STDGPU_LIB_DIR}
            LIBRARIES    ${STDGPU_LIBRARIES}
            DEPENDS      ext_stdgpu
        )
    endif()
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_stdgpu)
endif ()

# embree
if(USE_SYSTEM_EMBREE)
    find_package_3rdparty_library(3rdparty_embree
        PACKAGE embree
        TARGETS embree
    )
    if(NOT 3rdparty_embree_FOUND)
        set(USE_SYSTEM_EMBREE OFF)
    endif()
endif()
if(NOT USE_SYSTEM_EMBREE)
    include(${CloudViewer_3RDPARTY_DIR}/embree/embree.cmake)
    import_3rdparty_library(3rdparty_embree
        HIDDEN
        INCLUDE_DIRS ${EMBREE_INCLUDE_DIRS}
        LIB_DIR      ${EMBREE_LIB_DIR}
        LIBRARIES    ${EMBREE_LIBRARIES}
        DEPENDS      ext_embree
    )
endif()
list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_embree)

# WebRTC
if (BUILD_WEBRTC)
    # Incude WebRTC headers in CloudViewer.h.
    set(BUILD_WEBRTC_COMMENT "")

    # Build WebRTC from source for advanced users.
    option(BUILD_WEBRTC_FROM_SOURCE "Build WebRTC from source" OFF)
    mark_as_advanced(BUILD_WEBRTC_FROM_SOURCE)

    # WebRTC
    if (BUILD_WEBRTC_FROM_SOURCE)
        include(${CloudViewer_3RDPARTY_DIR}/webrtc/webrtc_build.cmake)
    else ()
        include(${CloudViewer_3RDPARTY_DIR}/webrtc/webrtc_download.cmake)
    endif ()
    import_3rdparty_library(3rdparty_webrtc
            HIDDEN
            INCLUDE_DIRS ${WEBRTC_INCLUDE_DIRS}
            LIB_DIR ${WEBRTC_LIB_DIR}
            LIBRARIES ${WEBRTC_LIBRARIES}
            DEPENDS ext_webrtc_all
            )
    target_link_libraries(3rdparty_webrtc INTERFACE 3rdparty_threads ${CMAKE_DL_LIBS})
    if (MSVC) # https://github.com/iimachines/webrtc-build/issues/2#issuecomment-503535704
        target_link_libraries(3rdparty_webrtc INTERFACE secur32 winmm dmoguids wmcodecdspuuid msdmo strmiids)
    endif ()
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_webrtc)

    # CivetWeb server
    include(${CloudViewer_3RDPARTY_DIR}/civetweb/civetweb.cmake)
    import_3rdparty_library(3rdparty_civetweb
            INCLUDE_DIRS ${CIVETWEB_INCLUDE_DIRS}
            LIB_DIR ${CIVETWEB_LIB_DIR}
            LIBRARIES ${CIVETWEB_LIBRARIES}
            DEPENDS ext_civetweb
            )
    list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_civetweb)
else ()
    # Don't incude WebRTC headers in CloudViewer.h.
    set(BUILD_WEBRTC_COMMENT "//")
endif ()

# ransacSD
include(${CloudViewer_3RDPARTY_DIR}/RANSAC_SD/RANSAC_SD.cmake)
import_3rdparty_library(3rdparty_ransacSD
        INCLUDE_DIRS ${RANSACSD_INCLUDE_DIRS}
        LIB_DIR ${RANSACSD_LIB_DIR}
        LIBRARIES ${RANSACSD_LIBRARIES}
        DEPENDS ext_ransacSD
        )
list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_ransacSD)


if (BUILD_RECONSTRUCTION)
    # freeimage
    if (WIN32)
        find_package(FreeImage QUIET)
        if (FREEIMAGE_FOUND)
            message(STATUS "FreeImage found in system")
        else ()
            message(STATUS "FreeImage not found in system and use prebuild")
            include(${CloudViewer_3RDPARTY_DIR}/freeimage/freeimage_build.cmake)
            import_3rdparty_library(3rdparty_freeimage
                    INCLUDE_DIRS ${FREEIMAGE_INCLUDE_DIRS}
                    LIB_DIR ${FREEIMAGE_LIB_DIR}
                    LIBRARIES ${EX_FREEIMAGE_LIBRARIES}
                    DEPENDS ext_freeimage
                    )
            add_dependencies(3rdparty_freeimage ext_freeimage)
            set(FREEIMAGE_TARGET "3rdparty_freeimage")
            list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_freeimage)
            # only for static freeimage usage
            # target_compile_definitions(3rdparty_freeimage INTERFACE FREEIMAGE_LIB)
        endif()
    else ()
        include(${CloudViewer_3RDPARTY_DIR}/freeimage/freeimage_build.cmake)
        import_shared_3rdparty_library(3rdparty_freeimage ext_freeimage
                INCLUDE_DIRS ${FREEIMAGE_INCLUDE_DIRS}
                LIB_DIR ${FREEIMAGE_LIB_DIR}
                LIBRARIES ${EX_FREEIMAGE_LIBRARIES}
                )
        add_dependencies(3rdparty_freeimage ext_freeimage)
        set(FREEIMAGE_TARGET "3rdparty_freeimage")
        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_freeimage)
    endif()

    # other dependency
    if (WIN32 OR APPLE)
        # gflags
        include(${CloudViewer_3RDPARTY_DIR}/gflags/gflags_build.cmake)
        import_3rdparty_library(3rdparty_gflags
                INCLUDE_DIRS ${GFLAGS_INCLUDE_DIRS}
                LIB_DIR ${GFLAGS_LIB_DIR}
                LIBRARIES ${EXT_GFLAGS_LIBRARIES}
                DEPENDS ext_gflags
                )
        set(GFLAGS_TARGET "3rdparty_gflags")
        if (WIN32)
            # fix gflags library bugs on windows
            target_link_libraries(3rdparty_gflags INTERFACE shlwapi.lib)
        endif()

        # glog
        include(${CloudViewer_3RDPARTY_DIR}/glog/glog_build.cmake)
        import_3rdparty_library(3rdparty_glog
                INCLUDE_DIRS ${GLOG_INCLUDE_DIRS}
                LIB_DIR ${GLOG_LIB_DIR}
                LIBRARIES ${EXT_GLOG_LIBRARIES}
                DEPENDS ext_glog
                )
        set(GLOG_TARGET "3rdparty_glog")
        add_dependencies(ext_glog ext_gflags)

        if (NOT WIN32) # alread done on suitesparse lib on windows
            # lapack and blas
            include(${CloudViewer_3RDPARTY_DIR}/lapack/lapack_build.cmake)
            import_3rdparty_library(3rdparty_lapack
                    LIB_DIR ${LAPACK_LIB_DIR}
                    LIBRARIES ${LAPACKBLAS_LIBRARIES}
                    DEPENDS ext_lapack
                    )
            set(LAPACK_TARGET "3rdparty_lapack")
            add_dependencies(3rdparty_lapack ext_lapack)
        endif()

        # suitesparse
        include(${CloudViewer_3RDPARTY_DIR}/suitesparse/suitesparse_build.cmake)
        import_3rdparty_library(3rdparty_suitesparse
                HIDDEN
                INCLUDE_DIRS ${SUITESPARSE_INCLUDE_DIRS}
                LIB_DIR ${SUITESPARSE_LIB_DIR}
                LIBRARIES ${EXT_SUITESPARSE_LIBRARIES}
                DEPENDS ext_suitesparse
                )
        if (NOT WIN32)
            add_dependencies(3rdparty_suitesparse 3rdparty_lapack)
        endif()

        # ceres
        include(${CloudViewer_3RDPARTY_DIR}/ceres-solver/ceres_build.cmake)
        import_3rdparty_library(3rdparty_ceres
                INCLUDE_DIRS ${CERES_INCLUDE_DIRS}
                LIB_DIR ${CERES_LIB_DIR}
                LIBRARIES ${EXT_CERES_LIBRARIES}
                DEPENDS ext_ceres
                )
        set(CERES_TARGET "3rdparty_ceres")
        add_dependencies(3rdparty_ceres 3rdparty_eigen3)
        add_dependencies(3rdparty_ceres 3rdparty_glog)
        add_dependencies(3rdparty_ceres 3rdparty_suitesparse)

        if (NOT WIN32) 
            add_dependencies(3rdparty_ceres 3rdparty_lapack)
            target_link_libraries(3rdparty_ceres INTERFACE 3rdparty_lapack)
        endif()
        target_link_libraries(3rdparty_ceres INTERFACE 3rdparty_eigen3 3rdparty_gflags 
                              3rdparty_glog 3rdparty_suitesparse)

        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_eigen3)
        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_ceres)
    elseif (UNIX)
        # Linux: boost must building from source due to _GLIBCXX_USE_CXX11_ABI=0 issues
        if (NOT ${GLIBCXX_USE_CXX11_ABI})
            include(${CloudViewer_3RDPARTY_DIR}/boost/boost.cmake)
            import_3rdparty_library(3rdparty_boost
                    INCLUDE_DIRS ${Boost_INCLUDE_DIR}
                    LIB_DIR ${Boost_LIBRARY_DIRS}
                    LIBRARIES ${BOOST_LIBRARIES}
                    DEPENDS ext_boost
                    )
            message(STATUS "BOOST_INCLUDE_DIR: " ${BOOST_INCLUDE_DIR})
            list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_boost)
        endif()

        # gflags
        include(${CloudViewer_3RDPARTY_DIR}/gflags/gflags_build.cmake)
        import_shared_3rdparty_library(3rdparty_gflags ext_gflags
                INCLUDE_DIRS ${GFLAGS_INCLUDE_DIRS}
                LIB_DIR ${GFLAGS_LIB_DIR}
                LIBRARIES ${EXT_GFLAGS_LIBRARIES}
                )
        set(GFLAGS_TARGET "3rdparty_gflags")
        add_dependencies(3rdparty_gflags ext_gflags)

        # glog
        include(${CloudViewer_3RDPARTY_DIR}/glog/glog_build.cmake)
        import_shared_3rdparty_library(3rdparty_glog ext_glog
                INCLUDE_DIRS ${GLOG_INCLUDE_DIRS}
                LIB_DIR ${GLOG_LIB_DIR}
                LIBRARIES ${EXT_GLOG_LIBRARIES}
                )
        set(GLOG_TARGET "3rdparty_glog")
        add_dependencies(3rdparty_glog ext_glog)
        add_dependencies(3rdparty_glog ext_gflags)

        # lapack and blas
        include(${CloudViewer_3RDPARTY_DIR}/lapack/lapack_build.cmake)
        import_shared_3rdparty_library(3rdparty_lapack ext_lapack
                INCLUDE_DIRS ${LAPACK_INCLUDE_DIRS}
                LIB_DIR ${LAPACK_LIB_DIR}
                LIBRARIES ${LAPACKBLAS_LIBRARIES}
                )
        set(LAPACK_TARGET "3rdparty_lapack")
        add_dependencies(3rdparty_lapack ext_lapack)

        # suitesparse
        include(${CloudViewer_3RDPARTY_DIR}/suitesparse/suitesparse_build.cmake)
        import_3rdparty_library(3rdparty_suitesparse
                INCLUDE_DIRS ${SUITESPARSE_INCLUDE_DIRS}
                LIB_DIR ${SUITESPARSE_LIB_DIR}
                LIBRARIES ${EXT_SUITESPARSE_LIBRARIES}
                DEPENDS ext_suitesparse
                )
        add_dependencies(ext_suitesparse ${LAPACK_TARGET})

        # ceres
        include(${CloudViewer_3RDPARTY_DIR}/ceres-solver/ceres_build.cmake)
        import_shared_3rdparty_library(3rdparty_ceres ext_ceres
                INCLUDE_DIRS ${CERES_INCLUDE_DIRS}
                LIB_DIR ${CERES_LIB_DIR}
                LIBRARIES ${EXT_CERES_LIBRARIES}
                )
        set(CERES_TARGET "3rdparty_ceres")
        add_dependencies(3rdparty_ceres ext_ceres)
        add_dependencies(3rdparty_ceres 3rdparty_eigen3)
        add_dependencies(3rdparty_ceres 3rdparty_gflags)
        add_dependencies(3rdparty_ceres 3rdparty_glog)
        add_dependencies(3rdparty_ceres 3rdparty_lapack)
        add_dependencies(3rdparty_ceres 3rdparty_suitesparse)

        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM "${CERES_TARGET}")
        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM "${GLOG_TARGET}")
        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM "${GFLAGS_TARGET}")
    endif ()
endif ()

# opencv
if (USE_SYSTEM_OPENCV)
    find_package_3rdparty_library(3rdparty_opencv
            PUBLIC
            PACKAGE OpenCV
            INCLUDE_DIRS OpenCV_INCLUDE_DIRS
            LIBRARIES OpenCV_LIBRARIES
            )
    if (3rdparty_opencv_FOUND)
        message(STATUS "Using system Opencv")
        message(STATUS "OpenCV version: ${3rdparty_opencv_VERSION}")
        if (NOT 3rdparty_opencv_VERSION VERSION_LESS "3.4.0")
            add_compile_definitions(HAVE_OPENCV3)
            message(STATUS "defined HAVE_OPENCV3")

            set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${3rdparty_opencv_INCLUDE_DIRS})
            include(CheckIncludeFileCXX)

            check_include_file_cxx(opencv2/ccalib/omnidir.hpp HAVE_OPENCV_CONTRIB)
            if (HAVE_OPENCV_CONTRIB)
                add_compile_definitions(HAVE_OPENCV_CONTRIB)
                set(WITH_OPENCV_CONTRIB ON)
            else ()
                message(STATUS "OPENCV_CONTRIB NOT PRESENT, DISABLING LOTS OF FUNCTIONALITY, SEE https://github.com/opencv/opencv_contrib")
            endif ()

            check_include_file_cxx(opencv2/xfeatures2d/nonfree.hpp HAVE_OPENCV_XFEATURES2D_NONFREE)
            if (HAVE_OPENCV_XFEATURES2D_NONFREE)
                add_compile_definitions(HAVE_OPENCV_XFEATURES2D_NONFREE)
            else ()
                message(STATUS "OPENCV_XFEATURES2D_NONFREE NOT PRESENT, DISABLING LOTS OF FUNCTIONALITY, SEE https://github.com/opencv/opencv_contrib")
            endif ()

            check_include_file_cxx(opencv2/cudafeatures2d.hpp HAVE_OPENCV_CUDAFEATURES2D)
            if (HAVE_OPENCV_CUDAFEATURES2D)
                add_compile_definitions(HAVE_OPENCV_CUDAFEATURES2D)
            else ()
                message(STATUS "OPENCV_CUDAFEATURES2D NOT PRESENT, DISABLING LOTS OF FUNCTIONALITY")
            endif ()

        else ()
            add_compile_definitions(HAVE_OPENCV2)
        endif ()

        # enable GPU enhanced SURF features
        # if BOTH CUDA and the OPENCV contrib cuda features are available
        if (CUDA_FOUND AND HAVE_OPENCV_CUDAFEATURES2D)
            add_compile_definitions(HAVE_CUDA)
            message(STATUS "defined HAVE_CUDA")
            set(CUDA_CUDART_LIBRARY_OPTIONAL ${CUDA_CUDART_LIBRARY})
        endif ()
    else ()
        set(USE_SYSTEM_OPENCV OFF)
        message(STATUS "Build Opencv from source")
    endif ()
endif ()
if (BUILD_OPENCV) # only needed by plugins: qAutoSeg and qManualSeg
    if (NOT USE_SYSTEM_OPENCV)
        include(${CloudViewer_3RDPARTY_DIR}/opencv/opencv_build.cmake)
        if (WIN32)
            import_3rdparty_library(3rdparty_opencv
                    INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS}
                    LIB_DIR ${OpenCV_LIB_DIR}
                    LIBRARIES ${OpenCV_LIBS}
                    DEPENDS ext_opencv
                    )
            # target_compile_definitions(3rdparty_opencv INTERFACE CV_STATIC_LIB)
        else ()
            import_shared_3rdparty_library(3rdparty_opencv ext_opencv
                INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS}
                LIB_DIR ${OpenCV_LIB_DIR}
                LIBRARIES ${OpenCV_LIBS}
            )
            add_dependencies(3rdparty_opencv ext_opencv)
        endif()
        # list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_opencv)
    else()
        # list(APPEND CloudViewer_3RDPARTY_PUBLIC_TARGETS_FROM_CUSTOM 3rdparty_opencv)
    endif()
endif()

# why compiling from source on windows
# main reason: use prebuild pcl and vtk with conda 
# which indeed not compiled vtk with qt support on windows
if (NOT USE_SYSTEM_VTK AND NOT USE_SYSTEM_PCL)
    find_package(Boost REQUIRED COMPONENTS
                 filesystem
                 iostreams)

    if (BUILD_VTK_FROM_SOURCE) # Note: building very slow
        message(STATUS "Building vtk from source")
        include(${CloudViewer_3RDPARTY_DIR}/vtk/vtk.cmake)
    else ()
        message(STATUS "Using pre-build vtk")
        include(${CloudViewer_3RDPARTY_DIR}/vtk/vtk_build_or_download.cmake)
    endif ()
    import_3rdparty_library(3rdparty_vtk
        INCLUDE_DIRS ${VTK_INCLUDE_DIRS}
        LIB_DIR ${VTK_LIBRARIES_DIRS}
        LIBRARIES ${VTK_LIBRARIES}
        DEPENDS ext_vtk
    )
    if(UNIX AND NOT APPLE)
        target_link_libraries(3rdparty_vtk INTERFACE ${CMAKE_DL_LIBS})
    endif()

    include(${CloudViewer_3RDPARTY_DIR}/pcl/pcl_build.cmake)
    import_3rdparty_library(3rdparty_pcl
        INCLUDE_DIRS ${PCL_INCLUDE_DIRS}
        LIB_DIR ${PCL_LIBRARY_DIRS}
        LIBRARIES ${PCL_LIBRARIES}
        DEPENDS ext_pcl
    )

    add_dependencies(3rdparty_pcl 3rdparty_vtk)
    target_link_libraries(3rdparty_pcl INTERFACE 3rdparty_vtk)
    target_link_libraries(3rdparty_pcl INTERFACE Boost::filesystem Boost::iostreams)
    list(APPEND CMAKE_MODULE_PATH "${CloudViewer_3RDPARTY_DIR}/CMake/pcl_cmake")
    find_package(FLANN REQUIRED)
    if (FLANN_FOUND)
        message(STATUS "FLANN_ROOT: ${FLANN_ROOT}")
        target_link_libraries(3rdparty_pcl INTERFACE FLANN::FLANN)
    endif()
    find_package(Qhull REQUIRED)
    if (QHULL_FOUND)
        message(STATUS "QHULL_ROOT: ${QHULL_ROOT}")
        target_link_libraries(3rdparty_pcl INTERFACE QHULL::QHULL)
    endif()
    # list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_vtk)
    # list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM 3rdparty_pcl)
else()
    set(3rdparty_vtk "")
    set(3rdparty_pcl "")
endif()

# Compactify list of external modules.
# This must be called after all dependencies are processed.
list(REMOVE_DUPLICATES CloudViewer_3RDPARTY_EXTERNAL_MODULES)

# Target linking order matters. When linking a target, the link libraries and
# include directories are set. If the order is wrong, it can cause missing
# symbols or wrong header versions. Generally, we want to link custom libs
# before system libs; the order among differnt libs can also matter.
#
# Current rules:
# 1. 3rdparty_curl should be linked before 3rdparty_png.
# 2. Link "FROM_CUSTOM" before "FROM_SYSTEM" targets.
# 3. ...
set(CloudViewer_3RDPARTY_PUBLIC_TARGETS
        ${CloudViewer_3RDPARTY_PUBLIC_TARGETS_FROM_CUSTOM}
        ${CloudViewer_3RDPARTY_PUBLIC_TARGETS_FROM_SYSTEM})
set(CloudViewer_3RDPARTY_HEADER_TARGETS
        ${CloudViewer_3RDPARTY_HEADER_TARGETS_FROM_CUSTOM}
        ${CloudViewer_3RDPARTY_HEADER_TARGETS_FROM_SYSTEM})
set(CloudViewer_3RDPARTY_PRIVATE_TARGETS
        ${CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM}
        ${CloudViewer_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM})
