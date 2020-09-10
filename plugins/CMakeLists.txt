cmake_minimum_required(VERSION 3.0)

project( ErowCloudViewerPlugins )

# reset buffers (for MacOS bundle)
set( EROWCLOUDVIEWER_PLUGINS CACHE INTERNAL "ErowCloudViewer plugin list" )

set( ErowCloudViewer_SOURCE_DIR "${CMAKE_SOURCE_DIR}/eCV" )
set( ErowCloudViewerPlugins_SOURCE_DIR "${CMAKE_SOURCE_DIR}/plugins" )

add_subdirectory( core )
add_subdirectory( example )
add_subdirectory( 3rdParty )
