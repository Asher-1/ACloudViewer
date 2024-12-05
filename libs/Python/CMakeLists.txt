cmake_minimum_required(VERSION 3.10)

if (BUILD_PYTHON_MODULE)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})

    include_directories(${CloudViewer_SOURCE_DIR})

    add_subdirectory(pybind)

    if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
        if (MSVC)
            SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
            SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)
            SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin/RelWithDebInfo)
        else ()
            SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
            SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        endif ()
    endif ()

    project(ECV_PYTHON_LIB)

    # CURRENT DIR
    file(GLOB header_list *.h)

    ##############################################################################################################
    ##############################################################################################################
    # UTILS PYBIND MODULE
    file(GLOB_RECURSE utility_pybind_list ${CMAKE_CURRENT_SOURCE_DIR}/utility/*)
    source_group("utility" FILES ${utility_pybind_list})
    # add_source_group(utility)
    file(GLOB recognition_pybind_list
            ${CMAKE_CURRENT_SOURCE_DIR}/recognition/*.h
            ${CMAKE_CURRENT_SOURCE_DIR}/recognition/*.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/recognition/*.hpp
            )
    source_group("recognition" FILES ${recognition_pybind_list})

    add_library(
            ${PROJECT_NAME} SHARED
            ${header_list}
            ${utility_pybind_list}
            ${recognition_pybind_list}
    )

    set_target_properties(${PROJECT_NAME} PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            )

    # dependence
    target_link_libraries(
        ${PROJECT_NAME}
        CVCoreLib
        ECV_DB_LIB
        Qt5::Core
        Qt5::PrintSupport
        pybind11::embed
        ${PYTHON_LIBRARIES}
    )

    # Add custom preprocessor definitions
    if (WIN32)
        # Do not treat includes from IMPORTED target as SYSTEM (Python headers in pybind11::embed).
        # This may be needed to resolve header conflicts, e.g. between Python release and debug headers.
        set_target_properties(${PROJECT_NAME} PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
    else ()
        # Do not treat includes from IMPORTED target as SYSTEM (Python headers in pybind11::embed).
        # This may be needed to resolve header conflicts, e.g. between Python release and debug headers.
        set_target_properties(${PROJECT_NAME} PROPERTIES NO_SYSTEM_FROM_IMPORTED ON COMPILE_DEFINITIONS ECV_PYTHON_LIB_LIBRARY_BUILD)
    endif ()

    target_compile_definitions(${PROJECT_NAME} PRIVATE ECV_PYTHON_LIB_LIBRARY_BUILD)
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_PYTHON_MODULE)

    if (NOT APPLE) # no need to copy targets on macos due to rpath
        # install (shared) lib to specified destinations
        InstallSharedLibrary(TARGET ${PROJECT_NAME})
    endif ()

    #import python dlls (if any, WIN32 only)
    include(cmake/ExportPythonDlls.cmake)
    export_python_dlls(${ACloudViewer_DEST_FOLDER})

endif ()

