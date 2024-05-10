option( PLUGIN_IO_QDRACO "Install qDracoIO plugin to read & write DRC files" OFF )

if( PLUGIN_IO_QDRACO )
	project( QDRACO_IO_PLUGIN )

	AddPlugin( NAME ${PROJECT_NAME} TYPE io )
	
	target_include_directories( ${PROJECT_NAME} PRIVATE ${DRACO_INCLUDE_DIRS} )
	target_link_libraries( ${PROJECT_NAME} ${DRACO_TARGET} )

	add_subdirectory( include )
	add_subdirectory( src )
	add_subdirectory( ui )
 
endif()
