cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QQUERYSPLAT "Install qQuerySplat (multi-view image-to-3DGS prediction GGML) plugin" OFF )

if( PLUGIN_STANDARD_QQUERYSPLAT )
    project( QQUERYSPLAT_PLUGIN )

    # AICore is required — skip plugin entirely if unavailable
    if(NOT TARGET AICore)
        message(WARNING "qQuerySplat: AICore target not available — skipping plugin. "
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
    # The plugin calls CVPluginAPI services directly (model catalog,
    # ecvTestDataRepository): keep the source-level dependency explicit.
    target_link_libraries(${PROJECT_NAME} CVPluginAPI)
    target_link_libraries(${PROJECT_NAME} Qt${QT_VERSION_MAJOR}::Concurrent)

    if(BUILD_UNIT_TESTS)
        add_subdirectory(tests)
    endif()
endif()
