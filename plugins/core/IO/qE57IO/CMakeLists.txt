
option( PLUGIN_IO_QE57 "Install qE57_IO plugin to read & write E57 files using libE57Format" OFF )

if ( PLUGIN_IO_QE57 )
	project( QE57_IO_PLUGIN )

	AddPlugin( NAME ${PROJECT_NAME} TYPE io )

	add_subdirectory( include )
	add_subdirectory( src )
	
	include( cmake/E57Support.cmake )
	target_link_LIBE57FORMAT( ${PROJECT_NAME} )
endif()
