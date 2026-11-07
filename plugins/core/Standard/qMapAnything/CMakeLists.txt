cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QMAPANYTHING "Install qMapAnything (universal feed-forward metric 3D reconstruction GGML) plugin" OFF )

if( PLUGIN_STANDARD_QMAPANYTHING )
    project( QMAPANYTHING_PLUGIN )

    # AICore is required — skip plugin entirely if unavailable
    if(NOT TARGET AICore)
        message(WARNING "qMapAnything: AICore target not available — skipping plugin. "
                        "Enable AICore_ENABLED and re-run CMake.")
        return()
    endif()

    AddPlugin( NAME ${PROJECT_NAME} )

    add_subdirectory( include )
    add_subdirectory( src )

    target_include_directories( ${PROJECT_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # Link to libAICore.so — ggml is PRIVATE inside AICore, not needed here
    target_link_libraries(${PROJECT_NAME} AICore)
    target_link_libraries(${PROJECT_NAME} 3rdparty_eigen3)
    target_link_libraries(${PROJECT_NAME} Qt${QT_VERSION_MAJOR}::Concurrent)

    if(BUILD_UNIT_TESTS)
        add_subdirectory(tests)
    endif()
endif()
