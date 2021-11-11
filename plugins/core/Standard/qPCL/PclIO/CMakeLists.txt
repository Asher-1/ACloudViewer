#CLOUDVIEWER  PCL I/O plugin (for PCD files)
project( QPCL_IO_PLUGIN )
AddPlugin( NAME ${PROJECT_NAME} TYPE io )

include_directories( ${QPCL_PLUGIN_ALGORIGHM_LIB_SOURCE_DIR}/Filters )
if (USE_PCL_BACKEND) 
    include_directories( ${QPCL_ENGINE_LIB_SOURCE_DIR} )
    include_directories( ${QPCL_ENGINE_LIB_SOURCE_DIR}/PclUtils )
endif()
include_directories( ${PCL_INCLUDE_DIRS} )

add_subdirectory( include )
add_subdirectory( src )
add_subdirectory( ui )

link_directories( ${PCL_LIBRARY_DIRS} )
if (NOT WIN32)
	add_definitions( ${PCL_DEFINITIONS} )
endif()
target_link_libraries(	${PROJECT_NAME} 
						QPCL_ENGINE_LIB
						QPCL_PLUGIN_ALGORIGHM_LIB)

#define the PCL_VER_1_6_OR_OLDER preprocessor to compile qPclIO with older versions of PCL
if ( PCL_VERSION VERSION_LESS  1.7 ) # VERSION_GREATER Works just like "greater or equal"
	target_compile_definitions( ${PROJECT_NAME} PUBLIC PCL_VER_1_6_OR_OLDER )
endif()
