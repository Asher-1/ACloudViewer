cmake_minimum_required(VERSION 3.0)

# https://cmake.org/cmake/help/git-stage/policy/CMP0074.html
#cmake_policy(SET CMP0074 OLD)

option( PLUGIN_STANDARD_QPCL "Check to install qPCL plugin" ON )

# ErowCloudViewer 'PCL' bridge plugin
if (PLUGIN_STANDARD_QPCL)
    include ( ${ECV_CMAKE_SCRIPTS_DIR}/CMakePolicies.cmake NO_POLICY_SCOPE )
	
	#ErowCloudViewer 'qPCLIO' I/O plugin (to load PCD files)
    add_subdirectory(PclIO)
    
    #PCL bridge library
    add_subdirectory(PclAlgorithms)
	
    project( QPCL_PLUGIN )
    
    #documentation
    add_subdirectory(doc)
    
	include_directories( ${PCL_INCLUDE_DIRS} )
	include_directories( ${QPCL_PLUGIN_ALGORIGHM_LIB_SOURCE_DIR} )
    include_directories( ${QPCL_PLUGIN_ALGORIGHM_LIB_BINARY_DIR} )
    
    include( ../../../CMakePluginTpl.cmake )
    
    target_link_libraries(${PROJECT_NAME} QPCL_PLUGIN_ALGORIGHM_LIB)
    
    link_directories( ${PCL_LIBRARY_DIRS} )
    add_definitions( ${PCL_DEFINITIONS} )
    
    #import PCL dlls (if any, WIN32 only)
    include( ExportPCLDlls.cmake )
    export_PCL_dlls( ${EROWCLOUDVIEWER_DEST_FOLDER} )
endif()
