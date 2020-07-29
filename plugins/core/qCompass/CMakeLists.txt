cmake_minimum_required(VERSION 3.0)

option( INSTALL_QCOMPASS_PLUGIN "Check to install compass plugin" OFF )

# CLOUDVIEWER  'compass' plugin
if (INSTALL_QCOMPASS_PLUGIN)
	project( QCOMPASS_PLUGIN )
	
	include ( ${ECV_CMAKE_SCRIPTS_DIR}/CMakePolicies.cmake NO_POLICY_SCOPE )

	# we need includes from the main CC source dir
	include_directories( ${ErowCloudViewer_SOURCE_DIR} )

	#we need ccOverlay classes
	file( GLOB ECV_PLUGIN_CUSTOM_HEADER_LIST ${ErowCloudViewer_SOURCE_DIR}/../common/ecvOverlayDialog*.h )
	file( GLOB ECV_PLUGIN_CUSTOM_SOURCE_LIST ${ErowCloudViewer_SOURCE_DIR}/../common/ecvOverlayDialog*.cpp )

	#we also need picking hub classes
	list( APPEND ECV_PLUGIN_CUSTOM_HEADER_LIST ${ErowCloudViewer_SOURCE_DIR}/../common/ecvPickingListener.h )
	list( APPEND ECV_PLUGIN_CUSTOM_HEADER_LIST ${ErowCloudViewer_SOURCE_DIR}/../common/ecvPickingHub.h )
	list( APPEND ECV_PLUGIN_CUSTOM_SOURCE_LIST ${ErowCloudViewer_SOURCE_DIR}/../common/ecvPickingHub.cpp )

	include_directories( src )

	add_subdirectory( src )
	add_subdirectory( ui )

	include( ../../CMakePluginTpl.cmake  )

endif()
