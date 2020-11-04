add_subdirectory(ManuallyAlignPointCloud)

macro(TOOL TOOL_NAME)
	include_directories( ${OPENGL_ENGINE_LIB_SOURCE_DIR} )
	if( MSVC )
	   include_directories( ${OPENGL_ENGINE_LIB_SOURCE_DIR}/msvc )
	endif()

	include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
	include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

	add_executable(${TOOL_NAME} "${TOOL_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${TOOL_NAME} ${DEPENDENCY})
    endforeach()
	
	target_link_libraries( ${TOOL_NAME} ECV_DB_LIB )
	target_link_libraries( ${TOOL_NAME} ECV_IO_LIB )
	
    set_target_properties(${TOOL_NAME} PROPERTIES FOLDER "Tools")
    ShowAndAbortOnWarning(${TOOL_NAME})
	cloudViewer_set_global_properties(${TOOL_NAME})
endmacro(TOOL)

TOOL(ConvertPointCloud)
TOOL(EncodeShader)
TOOL(GLInfo                     OPENGL_ENGINE_LIB ${GLFW_TARGET} ${OPENGL_TARGET})
TOOL(ManuallyCropGeometry 		OPENGL_ENGINE_LIB)
TOOL(MergeMesh  				OPENGL_ENGINE_LIB)
TOOL(ViewGeometry           	OPENGL_ENGINE_LIB)
