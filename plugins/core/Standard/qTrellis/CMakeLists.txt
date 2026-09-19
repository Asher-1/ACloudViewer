cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QTRELLIS "Install qTrellis (TRELLIS.2 image-to-3D GGML) plugin" OFF )

if( PLUGIN_STANDARD_QTRELLIS )
    project( QTRELLIS_PLUGIN )

    # AICore is required — skip plugin entirely if unavailable
    if(NOT TARGET AICore)
        message(WARNING "qTrellis: AICore target not available — skipping plugin. "
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

    # Qt::Sql is guaranteed by root find_package (CMakeExternalLibs.cmake)
    target_link_libraries(${PROJECT_NAME} Qt::Sql)
    # Qt::OpenGL (also in the root QT_COMPONENTS): QOpenGLWidget for the
    # pipeline-step 3D viewer.
    target_link_libraries(${PROJECT_NAME} Qt::OpenGL)

    if(MSVC)
        target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)
    endif()

    if(BUILD_UNIT_TESTS)
        add_subdirectory(tests)
    endif()
endif()
