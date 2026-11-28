cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QWEVISONE "Install qWeVisOne (WeVisOne-VL unified visual perception GGML) plugin" OFF )

if( PLUGIN_STANDARD_QWEVISONE )
    project( QWEVISONE_PLUGIN )

    # AICore is required — skip plugin entirely if unavailable
    if(NOT TARGET AICore)
        message(WARNING "qWeVisOne: AICore target not available — skipping plugin. "
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

    # Link to libAICore.so — ggml is PRIVATE inside AICore, not needed here.
    # The plugin consumes only the public aicore/wevisone_capi.h surface.
    target_link_libraries(${PROJECT_NAME} AICore CVPluginAPI)

    if(MSVC)
        target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)
    endif()

    if(BUILD_UNIT_TESTS)
        add_subdirectory(tests)
    endif()
endif()