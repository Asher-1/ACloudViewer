option( PLUGIN_IO_QPDAL "Install qPDALIO plugin to read & write LAS files (deprecated, prefer using qLAS I/O now)" OFF )

if( PLUGIN_IO_QPDAL )
	message( WARNING "The qPDAL plugin is now deprecated, prefer using qLAS instead" )

	project( QPDAL_IO_PLUGIN )

	find_package( PDAL 1.6 REQUIRED CONFIG )

	AddPlugin( NAME ${PROJECT_NAME} TYPE io )
	
	add_subdirectory( include )
	add_subdirectory( src )
	add_subdirectory( ui )

	if( PDAL_VERSION LESS 2.0 )
		set( JSON_ROOT_DIR "" CACHE PATH "jsoncpp root dir (PDAL/vendor/jsoncpp/dist)" )

		if( NOT JSON_ROOT_DIR )
				message( FATAL_ERROR "jsoncpp root dir is not specified (JSON_ROOT_DIR)" )
		endif()

        # set (JSON_ROOT_DIR "${CloudViewer_3RDPARTY_DIR}/jsoncpp/include/json")

		target_include_directories( ${PROJECT_NAME} PRIVATE ${JSON_ROOT_DIR} )
	endif()
	
	target_link_libraries( ${PROJECT_NAME} ${PDAL_LIBRARIES} )

	if( WIN32 )
		if ( MSVC_VERSION GREATER_EQUAL 1900 ) # Visual Studio 2017
			target_compile_definitions( ${PROJECT_NAME} PRIVATE WIN32_LEAN_AND_MEAN )
		endif()

		set( PDAL_DLL_DIR ${PDAL_LIBRARY_DIRS}/../bin )
		
		file( GLOB PDAL_DLL_FILES ${PDAL_DLL_DIR}/pdal*.dll )
		set (ALL_DLL_FILES ${PDAL_DLL_FILES})
		file( GLOB GDAL_DLL_FILES ${PDAL_DLL_DIR}/gdal*.dll )
		
		list( APPEND ALL_DLL_FILES ${GDAL_DLL_FILES})
		cloudViewer_install_files( "${ALL_DLL_FILES}" "${ACloudViewer_DEST_FOLDER}" )

		unset( ALL_DLL_FILES )
		unset( PDAL_DLL_DIR )
	endif()
endif()
