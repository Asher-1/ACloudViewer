cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QCORK "Install qCORK plugin" OFF )

# ACloudViewer Mesh CSG operations plugin (based on the Cork library)
if( PLUGIN_STANDARD_QCORK )
	project( QCORK_PLUGIN )
	
	AddPlugin( NAME ${PROJECT_NAME} )
	
	# Cork
	# Cork lib and MPIR/GMP are needed to build this plugin
	include(cmake/cork_download.cmake)
	include(cmake/CorkSupport.cmake)
	
	# Cork dependency - ensure it's built before the plugin
	add_dependencies(${PROJECT_NAME} ext_cork)
	
	add_subdirectory( include )
	add_subdirectory( src )
	add_subdirectory( ui )
	
	# Use CorkSupport.cmake function to link Cork library
	target_link_cork(${PROJECT_NAME})
endif()
