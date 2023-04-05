option(PLUGIN_STANDARD_QCANUPO "Check to install qCanupo plugin" OFF)

if (APPLE AND PLUGIN_STANDARD_QCANUPO)
    set(PLUGIN_STANDARD_QCANUPO OFF)
    message("PLUGIN QCANUPO is not supported on macos...")
endif ()

if (PLUGIN_STANDARD_QCANUPO)
    project(QCANUPO_PLUGIN)

    AddPlugin(NAME ${PROJECT_NAME})

    include(cmake/dlib_build.cmake)
    import_3rdparty_library(3rdparty_dlib
            INCLUDE_DIRS ${DLIB_INCLUDE_DIRS}
            LIB_DIR ${DLIB_LIB_DIR}
            LIBRARIES ${DLIB_LIBRARIES}
            DEPENDS ext_dlib
            )

    add_dependencies(${PROJECT_NAME} 3rdparty_dlib)
    option(QCANUPO_COMPILE_PRIVATE_METHODS "Compile qCanupo plugin with private methods (CNRS/UEB only)" OFF)
    if (QCANUPO_COMPILE_PRIVATE_METHODS)
        target_compile_definitions(${PROJECT_NAME} PRIVATE COMPILE_PRIVATE_CANUPO)
    endif ()

    add_subdirectory(include)
    add_subdirectory(src)
    add_subdirectory(ui)

    target_link_libraries(${PROJECT_NAME} 3rdparty_dlib)
endif ()
