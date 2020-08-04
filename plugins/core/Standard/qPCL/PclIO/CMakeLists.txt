cmake_minimum_required(VERSION 3.0)

#CLOUDVIEWER  PCL I/O plugin (for PCD files)
project( QPCL_IO_PLUGIN )

include_directories( ${QPCL_PLUGIN_ALGORIGHM_LIB_SOURCE_DIR}/Filters )
if (USE_PCL_BACKEND) 
	include_directories( ${QPCL_ENGINE_LIB_SOURCE_DIR}/PclUtils )
endif()
include_directories( ${PCL_INCLUDE_DIRS} )

set( CC_IS_IO_PLUGIN 1 )

include( ../../../../CMakePluginTpl.cmake )

target_link_libraries(${PROJECT_NAME} QPCL_PLUGIN_ALGORIGHM_LIB)
link_directories( ${PCL_LIBRARY_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )
if (USE_PCL_BACKEND)
	target_link_libraries( ${PROJECT_NAME} QPCL_ENGINE_LIB )
endif()	

#define the PCL_VER_1_6_OR_OLDER preprocessor to compile qPclIO with older versions of PCL
if ( PCL_VERSION VERSION_LESS  1.7 ) # VERSION_GREATER Works just like "greater or equal"
	set_property( TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS PCL_VER_1_6_OR_OLDER )
endif()
