# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------

# Copy files to the specified directory and for the active configurations
function(cloudViewer_install_files) # 2 (or 3) arguments:
    # ARGV0 = files (if it's a list you have to provide the list alias quoted!)
    # ARGV1 = target (directory)
    # ARGV2 = 0 for release only install
    #         1 for both release and debug install (if available)
    #         2 for debug only install (if available)

    if ( WIN32 AND ${ARGC} LESS_EQUAL 2)
        message(WARNING "For Windows configurations, it's better to specify whether the file should be copied for both release only (0), release and debug (1) or debug only (2)")
    endif()

    if ( ${ARGC} LESS_EQUAL 2 OR NOT ${ARGV2} EQUAL 2)

        message(STATUS "Files: ${ARGV0} will be installed in ${ARGV1}" )

        if( WIN32 ) # Windows
            if( NOT CMAKE_CONFIGURATION_TYPES )
                install( FILES ${ARGV0} DESTINATION ${ARGV1} )
            else()
                install( FILES ${ARGV0} CONFIGURATIONS Release DESTINATION ${ARGV1} )
                install( FILES ${ARGV0} CONFIGURATIONS RelWithDebInfo DESTINATION ${ARGV1}_withDebInfo )
            endif()
        elseif() # macOS or Linux
            install( FILES ${ARGV0} DESTINATION ${ARGV1} USE_SOURCE_PERMISSIONS)
            return()
        endif()
    endif()

    if ( ${ARGC} GREATER 2 )
        if ( ${ARGV2} EQUAL 1 OR ${ARGV2} EQUAL 2 )
            if(  NOT APPLE AND CMAKE_CONFIGURATION_TYPES )
                message(STATUS "Files: ${ARGV0} will be installed in ${ARGV1}_debug" )
                if (WIN32)
                    install( FILES ${ARGV0} CONFIGURATIONS Debug DESTINATION ${ARGV1}_debug)
                else()
                    install( FILES ${ARGV0} CONFIGURATIONS Debug DESTINATION ${ARGV1}_debug USE_SOURCE_PERMISSIONS)
                endif()
            endif()
        endif()
    endif()

endfunction(cloudViewer_install_files)

# Extended 'install' command depending on the build configuration and OS
# 4 arguments:
#   - ARGV0 = signature
#   - ARGV1 = target (warning: one project or one file at a time)
#   - ARGV2 = base install destination (_debug or _withDebInfo will be automatically appended if multi-conf is supported)
#   - ARGV3 = install destination suffix (optional)
function(cloudViewer_install_ext)
    if ("${ARGV0}" STREQUAL "DIRECTORY")
        set(INSTALL_OPTIONS FILES_MATCHING PATTERN "*")
    else ()
        set(INSTALL_OPTIONS "")
    endif ()

    list(APPEND INSTALL_OPTIONS PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

    if (APPLE)
        install(${ARGV0} ${ARGV1} DESTINATION ${ARGV2}${ARGV3}
                ${INSTALL_OPTIONS}
                )
        return()
    endif ()

    if (NOT CMAKE_CONFIGURATION_TYPES)
        install(${ARGV0} ${ARGV1} DESTINATION ${ARGV2}${ARGV3}
                ${INSTALL_OPTIONS})
    else ()
        install(${ARGV0} ${ARGV1} CONFIGURATIONS Release DESTINATION ${ARGV2}${ARGV3}
                ${INSTALL_OPTIONS})
        install(${ARGV0} ${ARGV1} CONFIGURATIONS RelWithDebInfo DESTINATION ${ARGV2}_withDebInfo${ARGV3}
                ${INSTALL_OPTIONS})
        install(${ARGV0} ${ARGV1} CONFIGURATIONS Debug DESTINATION ${ARGV2}_debug${ARGV3}
                ${INSTALL_OPTIONS})
    endif ()
endfunction(cloudViewer_install_ext)

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
endfunction(Directories)

function(cloudViewer_install_targets trgt)
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
endfunction(cloudViewer_install_targets)