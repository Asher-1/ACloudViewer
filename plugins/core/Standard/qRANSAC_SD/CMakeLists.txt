cmake_minimum_required(VERSION 3.10)

option( PLUGIN_STANDARD_QRANSAC_SD "Install qRansac_SD (Shape Detection) plugin" OFF )

if( PLUGIN_STANDARD_QRANSAC_SD )
	project( QRANSAC_SD_PLUGIN )

	AddPlugin( NAME ${PROJECT_NAME} )

	add_subdirectory( include )
	add_subdirectory( src )
	add_subdirectory( ui )
	
	target_link_libraries( ${PROJECT_NAME} 3rdparty_ransacSD )
endif()
