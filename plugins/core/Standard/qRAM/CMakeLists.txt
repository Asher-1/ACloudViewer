cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QRAM "Install Recognize Anything (RAM/RAM++/Tag2Text) plugin" OFF )

if( PLUGIN_STANDARD_QRAM )
    project( QRAM_PLUGIN )

    # AICore is required — skip plugin entirely if unavailable
    if(NOT TARGET AICore)
        message(WARNING "qRAM: AICore target not available — skipping plugin. "
                        "Enable AICore_ENABLED and re-run CMake.")
        return()
    endif()

    AddPlugin( NAME ${PROJECT_NAME} )

    add_subdirectory( include )
    add_subdirectory( src )

    target_include_directories( ${PROJECT_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            # QtCompat.h owns the Qt5.12/Qt6 split-behavior constants used
            # by this plugin's sources.
            ${CMAKE_SOURCE_DIR}/core/include
    )

    # qRAM directly consumes the shared plugin API for model downloads,
    # sample-data caching, integrity checks, settings, and DB helpers.
    # AddPlugin also supplies CVPluginAPI as the standard plugin ABI closure;
    # keeping this direct edge documents the source-level dependency.
    target_link_libraries( ${PROJECT_NAME}
        AICore
        CVPluginAPI
        Qt${QT_VERSION_MAJOR}::Widgets
    )

    # Shared camera / video-file playback. video_base itself publicly links
    # CVPluginAPI, but qRAM keeps its own direct CVPluginAPI edge above because
    # the dialog uses the API independently of video playback.
    if( TARGET video_base )
        target_link_libraries( ${PROJECT_NAME} video_base )
    endif()

    if(MSVC)
        target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)
    endif()

    if(BUILD_UNIT_TESTS OR AICore_BUILD_TESTS)
        add_subdirectory(tests)
    endif()
endif()
