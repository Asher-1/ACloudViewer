option(PLUGIN_IO_QLAS "LAS IO plugin using laszip" OFF)

if (PLUGIN_IO_QLAS)
    project(QLAS_IO_PLUGIN)

    AddPlugin(NAME ${PROJECT_NAME} TYPE io)

    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

    find_package(LASzip MODULE QUIET)
    if (NOT LASzip_FOUND)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(LASzip REQUIRED laszip)
        if (LASzip_FOUND)
            message(STATUS "Found LASzip-${LASZip_VERSION} ${LASZIP_INCLUDE_DIR}: ${LASZIP_LIBRARY}")
        else ()
            message(FATAL_ERROR "Failed to find LASzip")
        endif ()
    else ()
        message(STATUS "Found LASzip ${LASZIP_INCLUDE_DIR}: ${LASZIP_LIBRARY}")
    endif ()

    target_link_libraries(${PROJECT_NAME} LASzip::LASzip)

    target_compile_definitions(${PROJECT_NAME} PRIVATE -DQT_FORCE_ASSERTS)
    if (WIN32)
        # fix error C2872: “byte”: undeinfinde sysmbols on windows
        target_compile_definitions(${PROJECT_NAME} PRIVATE _HAS_STD_BYTE=0)
    endif()

    add_subdirectory(include)
    add_subdirectory(src)
    add_subdirectory(ui)

    if (WIN32)
        cloudViewer_install_files("${LASZIP_DLL}" "${ACloudViewer_DEST_FOLDER}" 1)
    endif ()
endif ()
