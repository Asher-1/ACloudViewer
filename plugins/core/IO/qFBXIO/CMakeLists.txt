cmake_minimum_required( VERSION 3.0 )

option( PLUGIN_IO_QFBX "Install qFBXIO plugin to read & write AutoDesk FBX files" OFF )

if ( PLUGIN_IO_QFBX )
	project( QFBX_IO_PLUGIN )
	
	option( FBX_SDK_SHARED_MODE "Build in shared mode" ON )

	include ( ${ECV_CMAKE_SCRIPTS_DIR}/CMakePolicies.cmake NO_POLICY_SCOPE )

	set( CC_IS_IO_PLUGIN 1 )

	# FBX SDK
	set( FBX_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/FBX-2020.0.1/include" CACHE PATH "FBX SDK include directory" )
	set( FBX_SDK_LIBRARY_FILE "${CMAKE_CURRENT_SOURCE_DIR}/FBX-2020.0.1/lib/X64/release/libfbxsdk.lib" CACHE FILEPATH "FBX SDK static library file" )
	set( FBX_XML2_LIBRARY_FILE "${CMAKE_CURRENT_SOURCE_DIR}/FBX-2020.0.1/lib/X64/release/libxml2-md.lib" CACHE FILEPATH "FBX XML2 static library file (for the 2019 SDK only)" )
	set( FBX_ZLIB_LIBRARY_FILE "${CMAKE_CURRENT_SOURCE_DIR}/FBX-2020.0.1/lib/X64/release/zlib-md.lib" CACHE FILEPATH "FBX ZLIB static library file (for the 2019 SDK only)" )

	if( CMAKE_CONFIGURATION_TYPES )
		set( FBX_SDK_LIBRARY_FILE_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/FBX-2020.0.1/lib/X64/debug/libfbxsdk.lib" CACHE FILEPATH "FBX SDK static debug library file" )
		set( FBX_XML2_LIBRARY_FILE_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/FBX-2020.0.1/lib/X64/debug/libxml2-md.lib" CACHE FILEPATH "FBX XML2 static debug library file (for the 2019 SDK only)" )
		set( FBX_ZLIB_LIBRARY_FILE_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/FBX-2020.0.1/lib/X64/debug/zlib-md.lib" CACHE FILEPATH "FBX ZLIB static debug library file (for the 2019 SDK only)" )
	endif()

	if ( NOT FBX_SDK_INCLUDE_DIR )
		message( SEND_ERROR "No FBX SDK include dir specified (FBX_SDK_INCLUDE_DIR)" )
	else()
		include_directories( ${FBX_SDK_INCLUDE_DIR} )
		if ( FBX_SDK_SHARED_MODE )
			add_definitions( -DFBXSDK_SHARED  )
		endif()
	endif()
	
	include_directories( src )

	add_subdirectory( src )
	
	include( ../../../CMakePluginTpl.cmake )
	
	# Link FBX
	# Release
	if( FBX_SDK_LIBRARY_FILE )
		if ( CMAKE_CONFIGURATION_TYPES )
			target_link_libraries( ${PROJECT_NAME} optimized ${FBX_SDK_LIBRARY_FILE} )
		else()
			target_link_libraries( ${PROJECT_NAME} ${FBX_SDK_LIBRARY_FILE} )
		endif()
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
