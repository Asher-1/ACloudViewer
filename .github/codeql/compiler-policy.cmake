# Only for explicitly enabled CodeQL scans; production optimization stays unchanged.
include_guard(GLOBAL)

if(CMAKE_VERSION VERSION_LESS 3.19)
    message(FATAL_ERROR "CodeQL compiler policy requires CMake 3.19 or newer")
endif()
if(NOT CLOUDVIEWER_CODEQL_OPTIMIZATION MATCHES "^O(1|3)$")
    message(FATAL_ERROR "CodeQL optimization must be O1 or O3")
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "CodeQL compiler policy requires a Linux build")
endif()
foreach(language C CXX)
    if(CMAKE_${language}_COMPILER_ID AND
       NOT CMAKE_${language}_COMPILER_ID MATCHES "^(GNU|Clang)$")
        message(FATAL_ERROR "CodeQL compiler policy requires GNU or Clang")
    endif()
endforeach()

# The main project auto-detects ccache; scans must compile rather than replay cached objects.
set(CMAKE_C_COMPILER_LAUNCHER "")
set(CMAKE_CXX_COMPILER_LAUNCHER "")

# O(directories + targets), retaining only the current directory's lists; source, link and export properties stay unchanged.
function(cloudviewer_codeql_apply_options directory)
    # Draco defaults to Release after project(); validation must also wait until configuration ends.
    if(directory STREQUAL CMAKE_SOURCE_DIR AND NOT CMAKE_BUILD_TYPE STREQUAL "Release")
        message(FATAL_ERROR "CodeQL compiler policy requires a Release build")
    endif()
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        get_target_property(type "${target}" TYPE)
        if(type MATCHES "^(EXECUTABLE|STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY)$")
            # Append target options at configuration end to override upstream Release flag resets, including Embree's.
            # Preserve NDEBUG, ISA, PIC, OpenMP and numerical options; only select the code-generation optimization level.
            target_compile_options("${target}" PRIVATE
                "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:-$CACHE{CLOUDVIEWER_CODEQL_OPTIMIZATION}>")
            set_target_properties("${target}" PROPERTIES
                C_COMPILER_LAUNCHER "" CXX_COMPILER_LAUNCHER "")
        endif()
    endforeach()
    get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(child IN LISTS children)
        cloudviewer_codeql_apply_options("${child}")
    endforeach()
endfunction()

# project() runs before upstream flag resets and target creation; defer until root configuration ends.
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
    CALL cloudviewer_codeql_apply_options "${CMAKE_SOURCE_DIR}")
