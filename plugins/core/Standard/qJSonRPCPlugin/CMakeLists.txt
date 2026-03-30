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

    add_subdirectory( include )
    add_subdirectory( src )
 
    target_include_directories( ${PROJECT_NAME}
                                PRIVATE
                                ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Optional PCV (ShadeVis) integration: link against QPCV_PLUGIN if built
    if ( TARGET QPCV_PLUGIN )
        target_link_libraries( ${PROJECT_NAME} QPCV_PLUGIN )
        target_include_directories( ${PROJECT_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/plugins/core/Standard/qPCV/include )
        target_compile_definitions( ${PROJECT_NAME} PRIVATE HAS_PCV_PLUGIN )
    endif()

endif()
