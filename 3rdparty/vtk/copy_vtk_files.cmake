file(GLOB VTK_FILES "${VTK_BINARY_DIR}/vtk*")

set(CONFIG_NAME "${CONFIG}")
# Ensure the destination directory exists before copying. Under parallel
# builds (MSBuild /m) the first project writing into
# CMAKE_RUNTIME_OUTPUT_DIRECTORY may not have run yet; copy_if_different
# with a directory target fails when the target directory does not exist,
# silently dropping every VTK DLL and producing wheels that fail to import
# with WinError 126. See 6_Build wheel CI log (ext_vtk POST_BUILD, 318
# "Error copying file (if different)" entries).
if (WIN32)
    file(MAKE_DIRECTORY "${DESTINATION_DIR}/${CONFIG_NAME}")
else ()
    file(MAKE_DIRECTORY "${DESTINATION_DIR}")
endif ()
foreach(FILE IN LISTS VTK_FILES)
		if (WIN32)
			execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${FILE}" "${DESTINATION_DIR}/${CONFIG_NAME}/"
                            RESULT_VARIABLE COPY_RESULT)
		else ()
			execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${FILE}" "${DESTINATION_DIR}/"
                            RESULT_VARIABLE COPY_RESULT)
		endif ()
        if (NOT COPY_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to copy VTK DLL '${FILE}' (exit ${COPY_RESULT}). "
                    "Target directory '${DESTINATION_DIR}/${CONFIG_NAME}' must exist and be writable.")
        endif ()
endforeach()