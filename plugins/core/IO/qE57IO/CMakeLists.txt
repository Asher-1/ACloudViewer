
option(PLUGIN_IO_QE57 "Install qE57_IO plugin to read & write E57 files using libE57Format" OFF)

if (PLUGIN_IO_QE57)
    project(QE57_IO_PLUGIN)

    AddPlugin(NAME ${PROJECT_NAME} TYPE io)

    add_subdirectory(include)
    add_subdirectory(src)

    include(extern/libE57Format.cmake)
    build_3rdparty_library(3rdparty_libE57Format DIRECTORY ${LIBE57FORMAT_SOURCE_DIR} DEPENDS ext_libE57Format)
    message(STATUS "libE57Format source dir: ${LIBE57FORMAT_SOURCE_DIR}")
#    add_subdirectory(${LIBE57FORMAT_SOURCE_DIR} EXCLUDE_FROM_ALL)

    set_target_properties(3rdparty_libE57Format PROPERTIES
            AUTOMOC OFF
            AUTORCC OFF
            AUTOUIC OFF
            )

    target_link_libraries(${PROJECT_NAME} 3rdparty_libE57Format)

    if (USING_STATIC_XERCES)
        target_compile_definitions(${PROJECT_NAME} PRIVATE XERCES_STATIC_LIBRARY)
    endif ()
endif ()
