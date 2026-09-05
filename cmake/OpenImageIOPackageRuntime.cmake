# Source-built OpenImageIO deployment for independently installable products.
# Callers provide the component data directory, app name, and ExternalProject
# prefix. The functions intentionally depend only on normal PostInstall
# variables so they can also be exercised by a standalone package smoke test.

function(oiio_runtime_dependency_is_system dependency result_variable)
    set(_oiio_is_system FALSE)
    if(APPLE)
        if(dependency MATCHES "^/System/Library/" OR
           dependency MATCHES "^/usr/lib/")
            set(_oiio_is_system TRUE)
        endif()
    elseif(WIN32)
        if(dependency MATCHES "^[A-Za-z]:[/\\\\]Windows[/\\\\]")
            set(_oiio_is_system TRUE)
        endif()
    else()
        get_filename_component(_oiio_dependency_name "${dependency}" NAME)
        if((dependency MATCHES "^/lib" OR dependency MATCHES "^/usr/lib") AND
           (_oiio_dependency_name MATCHES "^(libc|libdl|libgcc_s|libm|libpthread|librt|libstdc\\+\\+)[.]" OR
            _oiio_dependency_name MATCHES "^ld-linux"))
            set(_oiio_is_system TRUE)
        endif()
    endif()
    set(${result_variable} ${_oiio_is_system} PARENT_SCOPE)
endfunction()

function(deploy_oiio_runtime app_data_dir app_name external_install_dir)
    if(NOT BUILD_RECONSTRUCTION)
        return()
    endif()
    if(NOT IS_DIRECTORY "${external_install_dir}")
        message(FATAL_ERROR
            "OpenImageIO deployment requires source prefix ${external_install_dir}")
    endif()
    if(APPLE)
        set(_oiio_source_dir
            "${external_install_dir}/${CloudViewer_INSTALL_LIB_DIR}")
        set(_oiio_destination_dir
            "${app_data_dir}/${app_name}.app/Contents/Frameworks")
        file(GLOB _oiio_runtime "${_oiio_source_dir}/libOpenImageIO*.dylib")
        list(FILTER _oiio_runtime INCLUDE REGEX
             "/libOpenImageIO(\\.[0-9]+)*\\.dylib$")
        file(GLOB _oiio_util_runtime
             "${_oiio_source_dir}/libOpenImageIO_Util*.dylib")
    elseif(WIN32)
        set(_oiio_source_dir "${external_install_dir}/bin")
        set(_oiio_destination_dir "${app_data_dir}/lib")
        file(GLOB _oiio_runtime "${_oiio_source_dir}/OpenImageIO*.dll")
        list(FILTER _oiio_runtime INCLUDE REGEX "/OpenImageIO\\.dll$")
        file(GLOB _oiio_util_runtime
             "${_oiio_source_dir}/OpenImageIO_Util*.dll")
    else()
        set(_oiio_source_dir
            "${external_install_dir}/${CloudViewer_INSTALL_LIB_DIR}")
        set(_oiio_destination_dir "${app_data_dir}/${CloudViewer_INSTALL_LIB_DIR}")
        file(GLOB _oiio_runtime "${_oiio_source_dir}/libOpenImageIO.so*")
        file(GLOB _oiio_util_runtime
             "${_oiio_source_dir}/libOpenImageIO_Util.so*")
    endif()
    if(NOT _oiio_runtime OR NOT _oiio_util_runtime)
        message(FATAL_ERROR
            "OpenImageIO source build is missing its runtime closure under "
            "${_oiio_source_dir}")
    endif()
    file(MAKE_DIRECTORY "${_oiio_destination_dir}")
    file(COPY ${_oiio_runtime} ${_oiio_util_runtime}
         DESTINATION "${_oiio_destination_dir}" USE_SOURCE_PERMISSIONS
         FOLLOW_SYMLINK_CHAIN)

    # The OIIO ABI is pinned, but its enabled image/color libraries can be
    # shared. Resolve the actual closure after the source build rather than
    # maintaining a guessed PNG/JPEG/OpenEXR list per platform.
    file(GET_RUNTIME_DEPENDENCIES
         LIBRARIES ${_oiio_runtime} ${_oiio_util_runtime}
         DIRECTORIES "${_oiio_source_dir}"
         RESOLVED_DEPENDENCIES_VAR _oiio_resolved_dependencies
         UNRESOLVED_DEPENDENCIES_VAR _oiio_unresolved_dependencies)
    if(_oiio_unresolved_dependencies)
        list(JOIN _oiio_unresolved_dependencies ", " _oiio_unresolved_message)
        message(FATAL_ERROR
            "OpenImageIO runtime closure has unresolved dependencies: "
            "${_oiio_unresolved_message}")
    endif()

    set(_oiio_payload_libraries ${_oiio_runtime} ${_oiio_util_runtime})
    foreach(_oiio_dependency IN LISTS _oiio_resolved_dependencies)
        oiio_runtime_dependency_is_system("${_oiio_dependency}"
                                          _oiio_dependency_is_system)
        if(_oiio_dependency_is_system)
            continue()
        endif()
        file(COPY "${_oiio_dependency}" DESTINATION "${_oiio_destination_dir}"
             USE_SOURCE_PERMISSIONS FOLLOW_SYMLINK_CHAIN)
        list(APPEND _oiio_payload_libraries "${_oiio_dependency}")
    endforeach()

    if(APPLE)
        find_program(_oiio_install_name_tool install_name_tool REQUIRED)
        list(REMOVE_DUPLICATES _oiio_payload_libraries)
        foreach(_oiio_library IN LISTS _oiio_payload_libraries)
            get_filename_component(_oiio_library_name "${_oiio_library}" NAME)
            set(_oiio_deployed_library
                "${_oiio_destination_dir}/${_oiio_library_name}")
            if(NOT EXISTS "${_oiio_deployed_library}")
                continue()
            endif()
            execute_process(COMMAND "${_oiio_install_name_tool}"
                            -id "@rpath/${_oiio_library_name}"
                            "${_oiio_deployed_library}"
                            RESULT_VARIABLE _oiio_install_name_result)
            if(NOT _oiio_install_name_result EQUAL 0)
                message(FATAL_ERROR
                    "Could not set install name for ${_oiio_deployed_library}")
            endif()
            execute_process(COMMAND otool -L "${_oiio_deployed_library}"
                            OUTPUT_VARIABLE _oiio_linked_libraries
                            RESULT_VARIABLE _oiio_otool_result)
            if(NOT _oiio_otool_result EQUAL 0)
                message(FATAL_ERROR
                    "Could not inspect ${_oiio_deployed_library} with otool")
            endif()
            foreach(_oiio_dependency IN LISTS _oiio_payload_libraries)
                get_filename_component(_oiio_dependency_name
                                       "${_oiio_dependency}" NAME)
                # Do not rewrite an existing @rpath reference merely because
                # it has the same basename. Only absolute source-prefix
                # references need conversion into the package-local rpath.
                string(FIND "${_oiio_linked_libraries}"
                       "${_oiio_dependency} (" _oiio_dependency_index)
                if(NOT _oiio_dependency_index EQUAL -1)
                    execute_process(COMMAND "${_oiio_install_name_tool}"
                                    -change "${_oiio_dependency}"
                                    "@rpath/${_oiio_dependency_name}"
                                    "${_oiio_deployed_library}"
                                    RESULT_VARIABLE _oiio_change_result)
                    if(NOT _oiio_change_result EQUAL 0)
                        message(FATAL_ERROR
                            "Could not rewrite ${_oiio_dependency_name} in "
                            "${_oiio_deployed_library}")
                    endif()
                endif()
            endforeach()
        endforeach()
    endif()
endfunction()

function(verify_oiio_runtime_payload app_data_dir app_name)
    if(NOT BUILD_RECONSTRUCTION)
        return()
    endif()
    if(APPLE)
        set(_oiio_destination_dir
            "${app_data_dir}/${app_name}.app/Contents/Frameworks")
        file(GLOB _oiio_runtime
             "${_oiio_destination_dir}/libOpenImageIO.dylib"
             "${_oiio_destination_dir}/libOpenImageIO.[0-9]*.dylib")
        file(GLOB _oiio_util_runtime
             "${_oiio_destination_dir}/libOpenImageIO_Util*.dylib")
    elseif(WIN32)
        set(_oiio_destination_dir "${app_data_dir}/lib")
        file(GLOB _oiio_runtime "${_oiio_destination_dir}/OpenImageIO.dll")
        file(GLOB _oiio_util_runtime
             "${_oiio_destination_dir}/OpenImageIO_Util*.dll")
    else()
        set(_oiio_destination_dir
            "${app_data_dir}/${CloudViewer_INSTALL_LIB_DIR}")
        file(GLOB _oiio_runtime
             "${_oiio_destination_dir}/libOpenImageIO.so*")
        file(GLOB _oiio_util_runtime
             "${_oiio_destination_dir}/libOpenImageIO_Util.so*")
    endif()
    if(NOT _oiio_runtime OR NOT _oiio_util_runtime)
        message(FATAL_ERROR
            "Reconstruction package ${app_name} is missing the OpenImageIO "
            "runtime closure under ${_oiio_destination_dir}")
    endif()
    file(GET_RUNTIME_DEPENDENCIES
         LIBRARIES ${_oiio_runtime} ${_oiio_util_runtime}
         DIRECTORIES "${_oiio_destination_dir}"
         RESOLVED_DEPENDENCIES_VAR _oiio_verified_dependencies
         UNRESOLVED_DEPENDENCIES_VAR _oiio_unresolved_dependencies)
    if(_oiio_unresolved_dependencies)
        list(JOIN _oiio_unresolved_dependencies ", " _oiio_unresolved_message)
        message(FATAL_ERROR
            "Reconstruction package ${app_name} has an incomplete OpenImageIO "
            "runtime closure: ${_oiio_unresolved_message}")
    endif()
    file(REAL_PATH "${_oiio_destination_dir}" _oiio_destination_real)
    foreach(_oiio_dependency IN LISTS _oiio_verified_dependencies)
        oiio_runtime_dependency_is_system("${_oiio_dependency}"
                                          _oiio_dependency_is_system)
        if(_oiio_dependency_is_system)
            continue()
        endif()
        file(REAL_PATH "${_oiio_dependency}" _oiio_dependency_real)
        file(RELATIVE_PATH _oiio_dependency_relative
             "${_oiio_destination_real}" "${_oiio_dependency_real}")
        if(_oiio_dependency_relative MATCHES "^\\.\\.")
            message(FATAL_ERROR
                "Reconstruction package ${app_name} resolves OIIO dependency "
                "outside its payload: ${_oiio_dependency_real}")
        endif()
    endforeach()
endfunction()
