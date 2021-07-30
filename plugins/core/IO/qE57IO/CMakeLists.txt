
option(PLUGIN_IO_QE57 "Install qE57_IO plugin to read & write E57 files using libE57Format" OFF)

if (PLUGIN_IO_QE57)
    project(QE57_IO_PLUGIN)

    AddPlugin(NAME ${PROJECT_NAME} TYPE io)

    add_subdirectory(include)
    add_subdirectory(src)

    include(extern/xerces_build.cmake)
    import_3rdparty_library(3rdparty_xerces
            INCLUDE_DIRS ${XERCES_INCLUDE_DIRS}
            LIB_DIR ${XERCES_LIB_DIR}
            LIBRARIES ${XERCES_LIBRARIES}
            DEPENDS ext_xerces
            )

    include(extern/libE57Format.cmake)
    import_3rdparty_library(3rdparty_libE57Format
            INCLUDE_DIRS ${E57Format_INCLUDE_DIRS}
            LIB_DIR ${E57Format_LIB_DIR}
            LIBRARIES ${E57Format_LIBRARIES}
            DEPENDS ext_libE57Format
            )

    add_dependencies(${PROJECT_NAME} 3rdparty_xerces)
    add_dependencies(${PROJECT_NAME} 3rdparty_libE57Format)
    set_target_properties(3rdparty_libE57Format PROPERTIES
            AUTOMOC OFF
            AUTORCC OFF
            AUTOUIC OFF
            )

    #    find_package( XercesC REQUIRED )
    #target_link_libraries( ${PROJECT_NAME} XercesC::XercesC )
#    target_link_libraries(${PROJECT_NAME} ${XercesC_LIBRARY_RELEASE})
    target_link_libraries(${PROJECT_NAME} 3rdparty_xerces)
    target_link_libraries(${PROJECT_NAME} 3rdparty_libE57Format)

    if (USING_STATIC_XERCES)
        target_compile_definitions(${PROJECT_NAME} PRIVATE XERCES_STATIC_LIBRARY)
    endif ()
endif ()
