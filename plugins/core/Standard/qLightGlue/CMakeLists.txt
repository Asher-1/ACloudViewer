cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QLIGHTGLUE "Install qLightGlue (LightGlue feature matching) plugin" OFF )

if( PLUGIN_STANDARD_QLIGHTGLUE )
    project( QLIGHTGLUE_PLUGIN )

    if(NOT TARGET AICore)
        message(WARNING "qLightGlue: AICore target not available — skipping plugin. "
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

    if(NOT TARGET 3rdparty_opencv)
        message(WARNING "qLightGlue: BUILD_OPENCV=OFF — interactive SIFT matching "
                        "will be unavailable (matcher-only / Model Info still works).")
    endif()
    if(TARGET 3rdparty_opencv)
        target_link_libraries(${PROJECT_NAME} 3rdparty_opencv)
        target_compile_definitions(${PROJECT_NAME} PRIVATE QLIGHTGLUE_HAS_OPENCV)
    endif()

    target_link_libraries(${PROJECT_NAME} AICore Qt::Network)
endif()
