option( PLUGIN_IO_QFBX "Install qFBXIO plugin to read & write AutoDesk FBX files" OFF )

if ( PLUGIN_IO_QFBX )
	project( QFBX_IO_PLUGIN )
	
	AddPlugin( NAME ${PROJECT_NAME} TYPE io )
	
	option( FBX_SDK_SHARED_MODE "Build in shared mode" ON )
	
	# FBX
	include(cmake/FBX_download.cmake)
	import_3rdparty_library(3rdparty_fbx
        INCLUDE_DIRS ${FBX_DIR}/include/
        DEPENDS ext_fbx
        )
	add_dependencies(${PROJECT_NAME} 3rdparty_fbx)

	# FBX SDK
	set( FBX_SDK_LIBRARY_DIR "${FBX_DIR}/lib/X64" CACHE INTERNAL "" )
	set( FBX_SDK_INCLUDE_DIR "${FBX_DIR}/include" CACHE PATH "FBX SDK include directory" )
	message(STATUS "setting 'FBX_SDK_INCLUDE_DIR' to '${FBX_SDK_INCLUDE_DIR}'")
	message(STATUS "setting 'FBX_SDK_LIBRARY_DIR' to '${FBX_SDK_LIBRARY_DIR}'")
	set( FBX_SDK_LIBRARY_FILE "${FBX_SDK_LIBRARY_DIR}/release/libfbxsdk.lib" CACHE FILEPATH "FBX SDK static library file" )
	set( FBX_XML2_LIBRARY_FILE "${FBX_SDK_LIBRARY_DIR}/release/libxml2-md.lib" CACHE FILEPATH "FBX XML2 static library file (for the 2019 SDK only)" )
	set( FBX_ZLIB_LIBRARY_FILE "${FBX_SDK_LIBRARY_DIR}/release/zlib-md.lib" CACHE FILEPATH "FBX ZLIB static library file (for the 2019 SDK only)" )
    
	if( CMAKE_CONFIGURATION_TYPES )
		set( FBX_SDK_LIBRARY_FILE_DEBUG "${FBX_SDK_LIBRARY_DIR}/debug/libfbxsdk.lib" CACHE FILEPATH "FBX SDK static debug library file" )
		set( FBX_XML2_LIBRARY_FILE_DEBUG "${FBX_SDK_LIBRARY_DIR}/debug/libxml2-md.lib" CACHE FILEPATH "FBX XML2 static debug library file (for the 2019 SDK only)" )
		set( FBX_ZLIB_LIBRARY_FILE_DEBUG "${FBX_SDK_LIBRARY_DIR}/debug/zlib-md.lib" CACHE FILEPATH "FBX ZLIB static debug library file (for the 2019 SDK only)" )
	endif()

	if ( NOT FBX_SDK_INCLUDE_DIR )
		message( SEND_ERROR "No FBX SDK include dir specified (FBX_SDK_INCLUDE_DIR)" )
	else()
		include_directories( ${FBX_SDK_INCLUDE_DIR} )
		if ( FBX_SDK_SHARED_MODE )
			add_definitions( -DFBXSDK_SHARED  )
		endif()
	endif()
	
	add_subdirectory( include )
	add_subdirectory( src )
	
	# Link FBX
	# Release
	if( FBX_SDK_LIBRARY_FILE )
		if ( CMAKE_CONFIGURATION_TYPES )
			target_link_libraries( ${PROJECT_NAME} optimized ${FBX_SDK_LIBRARY_FILE} )
		else()
			target_link_libraries( ${PROJECT_NAME} ${FBX_SDK_LIBRARY_FILE} )
		endif()
		file( GLOB FBX_SDK_DLL_FILES ${FBX_SDK_LIBRARY_DIR}/release/libfbxsdk.dll )
		copy_files( "${FBX_SDK_DLL_FILES}" "${EROWCLOUDVIEWER_DEST_FOLDER}" )
	else()
		message( SEND_ERROR "FBX SDK library not found: can't link" )
	endif()
	
	if ( FBX_XML2_LIBRARY_FILE )
		if ( CMAKE_CONFIGURATION_TYPES )
			target_link_libraries( ${PROJECT_NAME} optimized ${FBX_XML2_LIBRARY_FILE} )
		else()
			target_link_libraries( ${PROJECT_NAME} ${FBX_XML2_LIBRARY_FILE} )
		endif()
	endif()

	if ( FBX_ZLIB_LIBRARY_FILE )
		if ( CMAKE_CONFIGURATION_TYPES )
			target_link_libraries( ${PROJECT_NAME} optimized ${FBX_ZLIB_LIBRARY_FILE} )
		else()
			target_link_libraries( ${PROJECT_NAME} ${FBX_ZLIB_LIBRARY_FILE} )
		endif()
	endif()

	# Debug
	if ( CMAKE_CONFIGURATION_TYPES )
		if ( FBX_SDK_LIBRARY_FILE_DEBUG )
			target_link_libraries( ${PROJECT_NAME} debug ${FBX_SDK_LIBRARY_FILE_DEBUG} )
		else()
			message( WARNING "No FBX SDK debug library file specified (FBX_SDK_LIBRARY_FILE_DEBUG)" )
		endif()
	
		if ( FBX_XML2_LIBRARY_FILE_DEBUG )
			target_link_libraries( ${PROJECT_NAME} debug ${FBX_XML2_LIBRARY_FILE_DEBUG} )
		endif()
		if ( FBX_ZLIB_LIBRARY_FILE_DEBUG )
			target_link_libraries( ${PROJECT_NAME} debug ${FBX_ZLIB_LIBRARY_FILE_DEBUG} )
		endif()
	endif()
endif()
