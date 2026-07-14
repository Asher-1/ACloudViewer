cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QDA3 "Install qDA3 (Depth Anything V3) plugin" OFF )

if( PLUGIN_STANDARD_QDA3 )
    if(NOT DA3_ENABLED OR NOT GGML_ENABLED)
        message(WARNING
            "qDA3: requires DA3_ENABLED and GGML_ENABLED. "
            "Reconfigure with -DDA3_ENABLED=ON -DGGML_ENABLED=ON "
            "(GGML is auto-enabled when DA3_ENABLED is ON).")
    endif()

    project( QDA3_PLUGIN )

    # DA3Core is required — skip plugin entirely if unavailable
    if(NOT TARGET DA3Core)
        message(WARNING "qDA3: DA3Core target not available — skipping plugin. "
                        "Enable DA3_ENABLED (requires GGML_ENABLED) and re-run CMake.")
        return()
    endif()

    AddPlugin( NAME ${PROJECT_NAME} )

    add_subdirectory( include )
    add_subdirectory( src )

    target_include_directories( ${PROJECT_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Link DA3Core (includes ggml transitively via PUBLIC linkage)
    target_link_libraries(${PROJECT_NAME} DA3Core Qt::Network)
endif()
