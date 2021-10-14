#add_subdirectory(ManuallyAlignPointCloud)

macro(cloudViewer_add_tool TOOL_NAME)
    include_directories( ${CloudViewer_SOURCE_DIR} )
    add_executable(${TOOL_NAME})

    target_sources(${TOOL_NAME} PRIVATE "${TOOL_NAME}.cpp")
    target_link_libraries(${TOOL_NAME} PRIVATE ${ARGN})

    target_link_libraries( ${TOOL_NAME} PRIVATE ECV_DB_LIB )
    target_link_libraries( ${TOOL_NAME} PRIVATE ECV_IO_LIB )

    # fix that You must build your code with position independent code if Qt was built with -reduce-relocations
    cloudViewer_set_targets_independent(${TOOL_NAME})
	
    set_target_properties(${TOOL_NAME} PROPERTIES FOLDER "Tools")
    cloudViewer_show_and_abort_on_warning(${TOOL_NAME})
    cloudViewer_set_global_properties(${TOOL_NAME})
endmacro()

cloudViewer_add_tool(ConvertPointCloud          CloudViewer )
cloudViewer_add_tool(GLInfo                     CloudViewer 3rdparty_glfw 3rdparty_opengl)
cloudViewer_add_tool(ManuallyCropGeometry       CloudViewer )
cloudViewer_add_tool(MergeMesh                  CloudViewer )
cloudViewer_add_tool(ViewGeometry           	CloudViewer )
cloudViewer_add_tool(HardwareResources          CloudViewer )
