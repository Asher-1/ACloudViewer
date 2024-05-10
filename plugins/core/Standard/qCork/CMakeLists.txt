option( PLUGIN_STANDARD_QCORK "Install qCORK plugin" OFF )

# ErowCloudViewer Mesh CSG operations plugin (based on the Cork library)
if( PLUGIN_STANDARD_QCORK )
	project( QCORK_PLUGIN )
	
	AddPlugin( NAME ${PROJECT_NAME} )
	
	# Cork
	# Cork lib and MPIR are needed to build this plugin
	include(cmake/cork_download.cmake)
	import_3rdparty_library(3rdparty_cork
			INCLUDE_DIRS ${CORK_DIR}/
			DEPENDS ext_cork
			)
	add_dependencies(${PROJECT_NAME} 3rdparty_cork)
	
	# Cork library (CC fork) https://github.com/cloudcompare/cork
	set( CORK_INCLUDE_DIR "${CORK_DIR}/cork" CACHE PATH "Cork include directory" )
	set( CORK_RELEASE_LIBRARY_FILE "${CORK_DIR}/cork/lib/x64/Release/wincork2013.lib" CACHE FILEPATH "Cork library file (release mode)" )
	if (WIN32)
		set( CORK_DEBUG_LIBRARY_FILE "${CORK_DIR}/cork/lib/x64/Debug/wincork2013.lib" CACHE FILEPATH "Cork library file (debug mode)" )
	endif()

	if ( NOT CORK_INCLUDE_DIR )
		message( SEND_ERROR "No Cork include dir specified (CORK_INCLUDE_DIR)" )
	else()
		include_directories( ${CORK_INCLUDE_DIR} )
	endif()

	set( MPIR_INCLUDE_DIR "${CORK_DIR}/mpir" CACHE PATH "MPIR include directory" )
	set( MPIR_RELEASE_LIBRARY_FILE "${CORK_DIR}/mpir/lib/x64/Release/mpir.lib" CACHE FILEPATH "MPIR library file (release mode)" )
	if (WIN32)
		set( MPIR_DEBUG_LIBRARY_FILE "${CORK_DIR}/mpir/lib/x64/Debug/mpir.lib" CACHE FILEPATH "MPIR library file (debug mode)" )
	endif()

	if ( NOT MPIR_INCLUDE_DIR )
		message( SEND_ERROR "No MPIR include dir specified (MPIR_INCLUDE_DIR)" )
	else()
		include_directories( ${MPIR_INCLUDE_DIR} )
	endif()
	
	add_subdirectory( include )
	add_subdirectory( src )
	add_subdirectory( ui )
	
	if( CORK_RELEASE_LIBRARY_FILE AND MPIR_RELEASE_LIBRARY_FILE )
		#Release mode only by default
		target_link_libraries( ${PROJECT_NAME} optimized ${CORK_RELEASE_LIBRARY_FILE} ${MPIR_RELEASE_LIBRARY_FILE} )
		
		#optional: debug mode
		if ( CORK_DEBUG_LIBRARY_FILE AND MPIR_DEBUG_LIBRARY_FILE )
			target_link_libraries( ${PROJECT_NAME} debug ${CORK_DEBUG_LIBRARY_FILE} ${MPIR_DEBUG_LIBRARY_FILE} )
		endif()
	else() #if ( NOT CORK_RELEASE_LIBRARY_FILE )
		message( SEND_ERROR "No Cork or MPIR release library files specified (CORK_RELEASE_LIBRARY_FILE / MPIR_RELEASE_LIBRARY_FILE)" )
	endif()
endif()
