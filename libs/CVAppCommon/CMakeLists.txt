cmake_minimum_required(VERSION 3.10)

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

if (OPTION_SUPPORT_GAMEPADS)
    find_package(Qt5
            COMPONENTS
            Gamepad
            REQUIRED
            )
    target_link_libraries(${PROJECT_NAME} Qt5::Gamepad)
endif ()

if (OPTION_SUPPORT_3DCONNEXION_DEVICES)
    target_link_3DXWARE(${PROJECT_NAME})
endif ()

if (NOT APPLE) # no need to copy targets on macos due to rpath
    # install (shared) lib to specified destinations
    InstallSharedLibrary(TARGET ${PROJECT_NAME})
endif ()
