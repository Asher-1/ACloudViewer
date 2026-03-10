cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

# https://cmake.org/cmake/help/git-stage/policy/CMP0074.html
#cmake_policy(SET CMP0074 OLD)

option( PLUGIN_STANDARD_QPCL "Check to install qPCL plugin" OFF )

# ACloudViewer 'PCL' bridge plugin
if (PLUGIN_STANDARD_QPCL)
    project( QPCL_PLUGIN )

    AddPlugin( NAME ${PROJECT_NAME} )

    include_directories( ${QVTK_ENGINE_LIB_SOURCE_DIR})
    include_directories( ${CMAKE_CURRENT_SOURCE_DIR})

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

    # PCL NURBS support (moved from VtkEngine)
    option(WITH_PCL_NURBS "Use PCL on_nurbs module or not" OFF)
    if (WITH_PCL_NURBS)
        target_compile_definitions(${PROJECT_NAME} PUBLIC WITH_PCL_NURBS)
    endif()

    # PCL version checks (moved from VtkEngine)
    if (PCL_VERSION VERSION_LESS 1.7)
        target_compile_definitions(${PROJECT_NAME} PUBLIC PCL_VER_1_6_OR_OLDER)
    endif ()
    if (PCL_VERSION VERSION_GREATER 1.7)
        target_compile_definitions(${PROJECT_NAME} PUBLIC LP_PCL_PATCH_ENABLED)
    endif ()

    #import PCL dlls (if any, WIN32 only)
    include(cmake/ExportPCLDlls.cmake)
    export_PCL_dlls(${ACloudViewer_DEST_FOLDER})

endif()

