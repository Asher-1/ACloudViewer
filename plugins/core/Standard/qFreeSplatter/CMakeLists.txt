cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QFREESPLATTER "Install qFreeSplatter (FreeSplatter 3D Gaussian) plugin" OFF )
option( PLUGIN_STANDARD_QFREESPLATTER_TOOLS "Build qFreeSplatter CLI (free_splatter-cli)" OFF )

if( PLUGIN_STANDARD_QFREESPLATTER )
    project( QFREESPLATTER_PLUGIN )

    # AICore is required — skip plugin entirely if unavailable
    if(NOT TARGET AICore)
        message(WARNING "qFreeSplatter: AICore target not available — skipping plugin. "
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
    target_link_libraries(${PROJECT_NAME} AICore Qt::Network)
    if(MSVC)
        target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)
    endif()
    if(TARGET CVAppCommon)
        target_link_libraries(${PROJECT_NAME} CVAppCommon)
    endif()

    # Face capture: enabled when OpenCV provides videoio + objdetect
    if(TARGET 3rdparty_opencv)
        target_link_libraries(${PROJECT_NAME} 3rdparty_opencv)
        target_compile_definitions(${PROJECT_NAME} PRIVATE HAS_OPENCV_FACE_CAPTURE)
        message(STATUS "qFreeSplatter: Face capture enabled (HAS_OPENCV_FACE_CAPTURE)")
    else()
        message(STATUS "qFreeSplatter: Face capture disabled (no 3rdparty_opencv)")
    endif()

    # SIBR in-app viewer: enabled only when qSIBR is built (PLUGIN_STANDARD_QSIBR=ON).
    # No link-time dependency on QSIBR_PLUGIN — runtime invoke via plugin manager.
    if(PLUGIN_STANDARD_QSIBR AND TARGET QSIBR_PLUGIN)
        target_compile_definitions(${PROJECT_NAME} PRIVATE HAS_QSIBR)
        message(STATUS "qFreeSplatter: SIBR visualization enabled (HAS_QSIBR)")
    else()
        message(STATUS "qFreeSplatter: SIBR visualization disabled (PLUGIN_STANDARD_QSIBR=OFF or qSIBR not built)")
    endif()

    if( PLUGIN_STANDARD_QFREESPLATTER_TOOLS )
        add_subdirectory( tools )
    endif()
endif()
