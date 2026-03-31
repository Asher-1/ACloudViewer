cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QPCV "Install qPCV plugin" OFF )

if( PLUGIN_STANDARD_QPCV )
	project( QPCV_PLUGIN )

	find_package( OpenGL REQUIRED )
	if( NOT OPENGL_FOUND )
		message( FATAL_ERROR "OpenGL required by PCV plugin" )
	endif()

	AddPlugin( NAME ${PROJECT_NAME} )

	# Export all symbols so that QJSON_RPC_PLUGIN can link PCV::Launch/GenerateRays
	set_property(TARGET ${PROJECT_NAME} PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)

	add_subdirectory( include )
	add_subdirectory( src )
	add_subdirectory( ui )

	target_include_directories( ${PROJECT_NAME} PRIVATE ${OpenGL_INCLUDE_DIR} )

	target_link_libraries( ${PROJECT_NAME} ${OPENGL_LIBRARIES} )
endif()
