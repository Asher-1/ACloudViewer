cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QDEEPLSD "Install qDeepLSD (DeepLSD wireframe) plugin" OFF )

if( PLUGIN_STANDARD_QDEEPLSD )
    project( QDEEPLSD_PLUGIN )

    if(NOT TARGET AICore)
        message(WARNING "qDeepLSD: AICore target not available — skipping plugin.")
        return()
    endif()

    AddPlugin( NAME ${PROJECT_NAME} )

    add_subdirectory( include )
    add_subdirectory( src )

    target_include_directories( ${PROJECT_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(${PROJECT_NAME} AICore Qt::Network)

    if(MSVC)
        target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)
    endif()
endif()
