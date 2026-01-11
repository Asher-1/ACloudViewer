cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})

project(CVAppCommon)

add_library(${PROJECT_NAME} SHARED)

add_subdirectory( devices )
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(ui)

target_link_libraries(${PROJECT_NAME}
        CVPluginAPI
        )

if (USE_PCL_BACKEND)
    include_directories(${QPCL_ENGINE_LIB_SOURCE_DIR})
    target_link_libraries(${PROJECT_NAME} QPCL_ENGINE_LIB)
endif ()

target_compile_definitions(${PROJECT_NAME} PRIVATE CVAPPCOMMON_LIBRARY_BUILD)

set_target_properties(${PROJECT_NAME} PROPERTIES
        AUTOUIC ON # FIXME Remove after everything has moved to targets and we can set it globally
        CXX_VISIBILITY_PRESET hidden
        )

# Add QDarkStyleSheet resources
target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/QDarkStyleSheet/qdarkstyle/dark/darkstyle.qrc
        ${CMAKE_CURRENT_LIST_DIR}/QDarkStyleSheet/qdarkstyle/light/lightstyle.qrc
)

if (OPTION_SUPPORT_GAMEPADS)
    if(USE_QT6)
        # Qt6 does not include Gamepad module - it was removed in Qt6
        # Gamepad support is disabled when using Qt6
        message(WARNING "Gamepad support is not available in Qt6. OPTION_SUPPORT_GAMEPADS will be ignored.")
        set(OPTION_SUPPORT_GAMEPADS OFF)
    else()
        find_package(Qt5
                COMPONENTS
                Gamepad
                REQUIRED
                )
        # Note: Gamepad is Qt5-only, so we use Qt5::Gamepad directly
        # (no unified Qt::Gamepad alias exists since it's not in Qt6)
        target_link_libraries(${PROJECT_NAME} Qt5::Gamepad)
    endif()
endif ()

if (OPTION_SUPPORT_3DCONNEXION_DEVICES)
    target_link_3DXWARE(${PROJECT_NAME})
endif ()

if (NOT APPLE) # no need to copy targets on macos due to rpath
    # install (shared) lib to specified destinations
    InstallSharedLibrary(TARGET ${PROJECT_NAME})
endif ()
