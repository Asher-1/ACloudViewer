cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

# ACloudViewer Json RPC plugin based on example for standard plugins

# Add an option to CMake to control whether we build this plugin or not
option( PLUGIN_STANDARD_QJSONRPC "Install Json RPC plugin" OFF )

if ( PLUGIN_STANDARD_QJSONRPC )
    project( QJSON_RPC_PLUGIN )

    AddPlugin( NAME ${PROJECT_NAME} )

    # Qt5/Qt6 compatibility - use unified Qt:: prefix
    # Note: Network and WebSockets are already found in CMakeExternalLibs.cmake
    target_link_libraries(${PROJECT_NAME} Qt::Core Qt::Network Qt::WebSockets)

    # Spirula Studio support (BUILD_SPIRULA): locator/launcher/importer shared
    # with qSpirula via spirula_base.
    if(TARGET spirula_base)
        target_link_libraries(${PROJECT_NAME} spirula_base)
    endif()

    # Selection-tools probes (view.selectRegion) drive cvViewSelectionManager
    # from the VTK engine, the same link layout the main app uses.
    if (TARGET QVTK_ENGINE_LIB)
        target_include_directories( ${PROJECT_NAME}
                                    PRIVATE
                                    ${QVTK_ENGINE_LIB_SOURCE_DIR}
                                    ${QVTK_ENGINE_LIB_SOURCE_DIR}/Tools/SelectionTools
        )
        target_link_libraries(${PROJECT_NAME} QVTK_ENGINE_LIB)
    endif()

    add_subdirectory( include )
    add_subdirectory( src )
 
    target_include_directories( ${PROJECT_NAME}
                                PRIVATE
                                ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
