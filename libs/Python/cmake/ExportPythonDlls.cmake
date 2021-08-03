function(export_python_dlls) # 1 argument: ARGV0 = destination directory

    #export python dlls (if any)
    if (WIN32 AND PYTHON_EXECUTABLE)

        # trim PYTHON_EXECUTABLE path if needed
        get_filename_component(ECV_PYTHON_DIR ${PYTHON_EXECUTABLE} PATH)
        set(PYTHON_DLL "${ECV_PYTHON_DIR}/python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.dll")
		copy_shared_library(${PROJECT_NAME}
			LIB_DIR      ${ECV_PYTHON_DIR}
			LIBRARIES    "python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")
        cloudViewer_install_files("${PYTHON_DLL}" "${ARGV0}")
        unset(PYTHON_DLL)

    endif ()

endfunction()
