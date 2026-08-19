cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QYOLO "Install qYOLO (YOLO object detection + metric depth GGML) plugin" OFF )

if( PLUGIN_STANDARD_QYOLO )
    project( QYOLO_PLUGIN )

    # AICore is required — skip plugin entirely if unavailable
    if(NOT TARGET AICore)
        message(WARNING "qYOLO: AICore target not available — skipping plugin. "
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

    # Shared camera / video-file playback (VideoPlaybackWidget + VideoFrameReader)
    if(TARGET video_base)
        target_link_libraries(${PROJECT_NAME} video_base)
    endif()

    # Qt::Sql is guaranteed by root find_package (CMakeExternalLibs.cmake)
    target_link_libraries(${PROJECT_NAME} Qt::Sql)

    if(TARGET 3rdparty_opencv)
        # HAS_OPENCV_FACE_CAPTURE is propagated PUBLIC by video_base
        target_link_libraries(${PROJECT_NAME} 3rdparty_opencv)
    endif()

    if(MSVC)
        target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)
    endif()

    if(BUILD_UNIT_TESTS)
        add_subdirectory(tests)
    endif()
endif()
