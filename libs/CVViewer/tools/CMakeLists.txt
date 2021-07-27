#add_subdirectory(ManuallyAlignPointCloud)

macro(cloudViewer_add_tool TOOL_NAME)
    include_directories( ${OPENGL_ENGINE_LIB_SOURCE_DIR} )
    add_executable(${TOOL_NAME})

    target_sources(${TOOL_NAME} PRIVATE "${TOOL_NAME}.cpp")
    target_link_libraries(${TOOL_NAME} PRIVATE ${ARGN})

    target_link_libraries( ${TOOL_NAME} PRIVATE ECV_DB_LIB )
    target_link_libraries( ${TOOL_NAME} PRIVATE ECV_IO_LIB )
	
    set_target_properties(${TOOL_NAME} PROPERTIES FOLDER "Tools")
    cloudViewer_show_and_abort_on_warning(${TOOL_NAME})
    cloudViewer_set_global_properties(${TOOL_NAME})
endmacro(TOOL)

cloudViewer_add_tool(ConvertPointCloud          OPENGL_ENGINE_LIB )
cloudViewer_add_tool(GLInfo                     OPENGL_ENGINE_LIB 3rdparty_glfw3 3rdparty_opengl)
cloudViewer_add_tool(ManuallyCropGeometry       OPENGL_ENGINE_LIB )
cloudViewer_add_tool(MergeMesh                  OPENGL_ENGINE_LIB )
cloudViewer_add_tool(ViewGeometry           	OPENGL_ENGINE_LIB )
cloudViewer_add_tool(HardwareResources          OPENGL_ENGINE_LIB )
