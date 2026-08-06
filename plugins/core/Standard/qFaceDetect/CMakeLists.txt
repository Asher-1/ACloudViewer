cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QFACEDETECT "Install qFaceDetect (face-detect.cpp GGML) plugin" OFF )

if( PLUGIN_STANDARD_QFACEDETECT )
    project( QFACEDETECT_PLUGIN )

    if(NOT TARGET AICore)
        message(WARNING "qFaceDetect: AICore target not available — skipping plugin.")
        return()
    endif()

    AddPlugin( NAME ${PROJECT_NAME} )

    add_subdirectory( include )
    add_subdirectory( src )

    target_include_directories( ${PROJECT_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(${PROJECT_NAME} AICore)

    # Qt::Sql is guaranteed by root find_package (CMakeExternalLibs.cmake)
    target_link_libraries(${PROJECT_NAME} Qt::Sql)

    if(TARGET 3rdparty_opencv)
        target_link_libraries(${PROJECT_NAME} 3rdparty_opencv)
        target_compile_definitions(${PROJECT_NAME} PRIVATE HAS_OPENCV_FACE_CAPTURE)
    endif()

    cloudviewer_plugin_link_zlib_minizip(${PROJECT_NAME}
        PLUGIN_NAME qFaceDetect)

    if(MSVC)
        target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)
    endif()

    if(BUILD_UNIT_TESTS)
        add_subdirectory(tests)
    endif()
endif()
