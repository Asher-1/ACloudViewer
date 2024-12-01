# ACloudViewer Json RPC plugin based on example for standard plugins

# Add an option to CMake to control whether we build this plugin or not
option( PLUGIN_STANDARD_QJSONRPC "Install Json RPC plugin" OFF )

if ( PLUGIN_STANDARD_QJSONRPC )
    project( QJSON_RPC_PLUGIN )

    AddPlugin( NAME ${PROJECT_NAME} )

    find_package(Qt5 COMPONENTS Network WebSockets REQUIRED)
    target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Network Qt5::WebSockets)

    add_subdirectory( include )
    add_subdirectory( src )
 
    target_include_directories( ${PROJECT_NAME}
                                PRIVATE
                                ${CMAKE_CURRENT_SOURCE_DIR}
    )

endif()
