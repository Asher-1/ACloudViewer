# Windows only
if (WIN32)
    option(PLUGIN_IO_QLAS_FWF "Check to install qLAS FWF I/O plugin" OFF)

    if (PLUGIN_IO_QLAS_FWF)
        #CloudViewer LAS FWF (i.e. version >= 1.3) I/O plugin
        project(QLAS_FWF_IO_PLUGIN)

        AddPlugin(NAME ${PROJECT_NAME} TYPE io)

        # LasLibZip
        include(cmake/LasLibZip_download.cmake)
        import_3rdparty_library(3rdparty_lasLibzip
                INCLUDE_DIRS ${LASLIBZIP_DIR}/
                DEPENDS ext_lasLibzip
                )
        add_dependencies(${PROJECT_NAME} 3rdparty_lasLibzip)

        add_subdirectory(include)
        add_subdirectory(src)
        add_subdirectory(ui)

        # Uses headers from the qPDALIO plugin...
        target_include_directories(${PROJECT_NAME}
                PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/../qPDALIO/include"
                )

        include(cmake/LASLibSupport.cmake)

        target_link_LASLib(${PROJECT_NAME})

    endif ()

endif ()
