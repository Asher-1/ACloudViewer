cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

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
        # fix error C2872: "byte": undeinfinde sysmbols on windows
        target_compile_definitions(${PROJECT_NAME} PRIVATE _HAS_STD_BYTE=0)
    endif()

    add_subdirectory(include)
    add_subdirectory(src)
    add_subdirectory(ui)

    if(BUILD_UNIT_TESTS)
        add_executable(test_las_vlr tests/test_las_vlr.cpp)
        target_include_directories(test_las_vlr PRIVATE include)
        if(NOT MSVC)
            target_compile_options(test_las_vlr PRIVATE -fPIC)
        endif()
        target_link_libraries(test_las_vlr PRIVATE CV_DB_LIB LASzip::LASzip gtest_main)
        cloudviewer_add_test(NAME test_las_vlr COMMAND test_las_vlr
            LABELS "unit;io;las;serialization")
        cloudviewer_set_test_runtime_environment(test_las_vlr)
        set_tests_properties(test_las_vlr PROPERTIES TIMEOUT 30)
    endif()

    if (WIN32)
        cloudViewer_install_files("${LASZIP_DLL}" "${ACloudViewer_DEST_FOLDER}" 1)
    endif ()
endif ()
