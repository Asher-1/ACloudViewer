cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

# https://cmake.org/cmake/help/git-stage/policy/CMP0074.html
#cmake_policy(SET CMP0074 OLD)

option( PLUGIN_STANDARD_QPCL "Check to install qPCL plugin" ON )

# ACloudViewer 'PCL' bridge plugin
if (PLUGIN_STANDARD_QPCL)
    project( QPCL_PLUGIN )

    AddPlugin( NAME ${PROJECT_NAME} )

    include_directories( ${QPCL_ENGINE_LIB_SOURCE_DIR})

    add_subdirectory( doc )
    add_subdirectory( include )
    add_subdirectory( src )

    #ACloudViewer 'qPCLIO' I/O plugin (to load PCD files)
    add_subdirectory(PclIO)
    
    #PCL bridge library
    add_subdirectory(PclAlgorithms)
    
    include_directories( ${PCL_INCLUDE_DIRS} )
    include_directories( ${QPCL_PLUGIN_ALGORIGHM_LIB_SOURCE_DIR} )
    include_directories( ${QPCL_PLUGIN_ALGORIGHM_LIB_BINARY_DIR} )
    
    target_link_libraries( ${PROJECT_NAME} QPCL_PLUGIN_ALGORIGHM_LIB)

endif()
