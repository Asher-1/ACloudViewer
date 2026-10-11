cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QRELATEANYTHING "Install RelateAnything scene relation plugin" OFF )

if( PLUGIN_STANDARD_QRELATEANYTHING )
    project( QRELATEANYTHING_PLUGIN )

    if(NOT TARGET AICore)
        message(WARNING "qRelateAnything needs AICore_ENABLED")
        return()
    endif()

    AddPlugin( NAME ${PROJECT_NAME} )

    add_subdirectory( include )
    add_subdirectory( src )

    target_include_directories( ${PROJECT_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # qRelateAnything directly consumes the shared plugin API for model downloads,
    # sample-data caching, integrity checks, settings, and DB helpers.
    # AddPlugin also supplies CVPluginAPI as the standard plugin ABI closure;
    # keeping this direct edge documents the source-level dependency.
    target_link_libraries( ${PROJECT_NAME}
        AICore
        CVPluginAPI
        Qt${QT_VERSION_MAJOR}::Widgets
    )

    # Shared camera / video-file playback. video_base itself publicly links
    # CVPluginAPI, but qRelateAnything keeps its own direct CVPluginAPI edge above because
    # the dialog uses the API independently of video playback.
    if( TARGET video_base )
        target_link_libraries( ${PROJECT_NAME} video_base )
    endif()

    if( BUILD_UNIT_TESTS OR AICore_BUILD_TESTS )
        add_subdirectory( tests )
    endif()
endif()
