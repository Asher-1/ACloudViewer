cmake_minimum_required(VERSION 3.0)

project( ErowCloudViewerPlugins )

# Add our cmake module path so we don't need relative paths for these
list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" )

include( Plugins )

# reset buffers (for MacOS bundle)
set( EROWCLOUDVIEWER_PLUGINS CACHE INTERNAL "ErowCloudViewer plugin list" )

set( ErowCloudViewer_SOURCE_DIR "${CMAKE_SOURCE_DIR}/eCV" )
set( ErowCloudViewerPlugins_SOURCE_DIR "${CMAKE_SOURCE_DIR}/plugins" )

add_subdirectory( core )
add_subdirectory( example )
add_subdirectory( 3rdParty )

# Check for any private plugins and add them
get_filename_component( PRIVATE_DIR private ABSOLUTE )

if( EXISTS "${PRIVATE_DIR}" )	
	file( GLOB subdirectories "${PRIVATE_DIR}/*" )
	
	foreach( dir ${subdirectories} )
	    if( IS_DIRECTORY ${dir} AND EXISTS ${dir}/CMakeLists.txt )
			# In case the developer generated the BUILD project in the same folder as the sources
			if( NOT ${dir} STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles" )
				message( STATUS "Found private plugin: " ${dir} )
				add_subdirectory( ${dir} )
			endif()
		endif()
	endforeach()
endif()