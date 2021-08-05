# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------

# Copy files to the specified directory and for the active configurations
function(cloudViewer_install_files) # 2 (or 3) arguments:
    # ARGV0 = files (if it's a list you have to provide the list alias quoted!)
    # ARGV1 = target (directory)
    # ARGV2 = 1 for debug install (if available)

    message(STATUS "Files: ${ARGV0} will be installed in ${ARGV1}")

    if (APPLE)
        install(FILES ${ARGV0} DESTINATION ${ARGV1})
        return()
    endif ()

    if (NOT CMAKE_CONFIGURATION_TYPES)
        install(FILES ${ARGV0} DESTINATION ${ARGV1})
    else ()
        install(FILES ${ARGV0} CONFIGURATIONS Release DESTINATION ${ARGV1})
        install(FILES ${ARGV0} CONFIGURATIONS RelWithDebInfo DESTINATION ${ARGV1}_withDebInfo)
        install(FILES ${ARGV0} CONFIGURATIONS Debug DESTINATION ${ARGV1}_debug)
        if (${ARGC} GREATER 2)
            if (${ARGV2} EQUAL 1)
                install(FILES ${ARGV0} CONFIGURATIONS Debug DESTINATION ${ARGV1}_debug)
            endif ()
        endif ()
    endif ()
endfunction(cloudViewer_install_files)

# Extended 'install' command depending on the build configuration and OS
# 4 arguments:
#   - ARGV0 = signature
#   - ARGV1 = target (warning: one project or one file at a time)
#   - ARGV2 = base install destination (_debug or _withDebInfo will be automatically appended if multi-conf is supported)
#   - ARGV3 = install destination suffix (optional)
function(cloudViewer_install_ext)
    if (APPLE)
        install(${ARGV0} ${ARGV1} DESTINATION ${ARGV2}${ARGV3})
        return()
    endif ()

    if (NOT CMAKE_CONFIGURATION_TYPES)
        install(${ARGV0} ${ARGV1} DESTINATION ${ARGV2}${ARGV3})
    else ()
        install(${ARGV0} ${ARGV1} CONFIGURATIONS Release DESTINATION ${ARGV2}${ARGV3})
        install(${ARGV0} ${ARGV1} CONFIGURATIONS RelWithDebInfo DESTINATION ${ARGV2}_withDebInfo${ARGV3})
        install(${ARGV0} ${ARGV1} CONFIGURATIONS Debug DESTINATION ${ARGV2}_debug${ARGV3})
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