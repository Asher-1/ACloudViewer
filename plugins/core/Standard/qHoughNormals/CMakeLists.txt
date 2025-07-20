cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QHOUGH_NORMALS "Install qHoughNormals plugin" OFF )

# ACloudViewer 'Hough Normals' plugin (see https://github.com/aboulch/normals_Hough)
if( PLUGIN_STANDARD_QHOUGH_NORMALS )
	project( QHOUGH_NORMALS_PLUGIN )

	AddPlugin( NAME ${PROJECT_NAME} )
	
	# add_subdirectory( extern/nanoflann )
	add_subdirectory( extern/normals_Hough )
	add_subdirectory( include )
	add_subdirectory( src )
	add_subdirectory( ui )
	
	target_link_libraries( ${PROJECT_NAME} 3rdparty_eigen3 3rdparty_nanoflann)
endif()