cmake_minimum_required(VERSION 3.0)

option( INSTALL_QCSV_MATRIX_IO_PLUGIN "Check to install qCSVMatrix I/O plugin" OFF )

# CLOUDVIEWER  CSV Matrix I/O plugin
if (INSTALL_QCSV_MATRIX_IO_PLUGIN)
    project( QCSV_MATRIX_IO_PLUGIN )
	
	include ( ${ECV_CMAKE_SCRIPTS_DIR}/CMakePolicies.cmake NO_POLICY_SCOPE )
    
    set( CC_IS_IO_PLUGIN 1 )
	
	include_directories( src )
	
	add_subdirectory( src )
	add_subdirectory( ui )
    
    include( ../../CMakePluginTpl.cmake )
endif()
