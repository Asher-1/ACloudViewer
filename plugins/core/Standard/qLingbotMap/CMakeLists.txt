cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QLINGBOTMAP "Install qLingbotMap (LingBot-Map streaming 3D reconstruction GGML) plugin" OFF )

if( PLUGIN_STANDARD_QLINGBOTMAP )
    project( QLINGBOTMAP_PLUGIN )

    # AICore is required — skip plugin entirely if unavailable
    if(NOT TARGET AICore)
        message(WARNING "qLingbotMap: AICore target not available — skipping plugin. "
                        "Enable AICore_ENABLED and re-run CMake.")
        return()
    endif()

    AddPlugin( NAME ${PROJECT_NAME} )

    add_subdirectory( include )
    add_subdirectory( src )

    target_include_directories( ${PROJECT_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Link to libAICore.so — ggml is PRIVATE inside AICore, not needed here
    target_link_libraries(${PROJECT_NAME} AICore)

    # Shared video infrastructure (OpenCVFrameSource decode backend +
    # cvMatToQImage); HAS_OPENCV_FACE_CAPTURE is propagated PUBLIC by
    # video_base, conditional on 3rdparty_opencv (BUILD_OPENCV).
    if(TARGET video_base)
        target_link_libraries(${PROJECT_NAME} video_base)
    endif()

    if(TARGET 3rdparty_opencv)
        target_link_libraries(${PROJECT_NAME} 3rdparty_opencv)
    else()
        message(WARNING "qLingbotMap: BUILD_OPENCV is OFF — the video-file "
                        "input is disabled (image-folder input is "
                        "unaffected). Set BUILD_OPENCV=ON for video support.")
    endif()

    if(MSVC)
        target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)
    endif()
endif()
