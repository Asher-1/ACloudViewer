cmake_minimum_required(VERSION 3.10)

if (BUILD_PYTHON_MODULE)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})

    include_directories(${OPENGL_ENGINE_LIB_SOURCE_DIR})
    if (MSVC)
        include_directories(${OPENGL_ENGINE_LIB_SOURCE_DIR}/msvc)
    endif ()

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

    set(CMAKE_CXX_VISIBILITY_PRESET hidden)

    # CURRENT DIR
    file(GLOB header_list *.h)

    ##############################################################################################################
    ##############################################################################################################
    # UTILS PYBIND MODULE
    file(GLOB_RECURSE utility_pybind_list ${CMAKE_CURRENT_SOURCE_DIR}/utility/*)
    source_group("utility" FILES ${utility_pybind_list})
    # ADD_SOURCE_GROUP(utility)
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

    # Qt
    if (NOT CMAKE_VERSION VERSION_LESS 3.0)
        target_link_libraries(
                ${PROJECT_NAME}
                pybind11::embed
                ${PYTHON_LIBRARIES}
                Qt5::Core
                Qt5::PrintSupport
                OPENGL_ENGINE_LIB
        )
    else ()
        target_include_directories(
                ${PROJECT_NAME}
                ${PYBIND11_INCLUDE_DIR}
                ${PYTHON_INCLUDE_DIRS}
        )

        target_compile_options(
                ${PROJECT_NAME}
                ${PYBIND11_CPP_STANDARD}
        )

        target_link_libraries(${PROJECT_NAME}
                pybind11::embed
                ${PYTHON_LIBRARIES}
                Qt5::Core
                Qt5::PrintSupport
                OPENGL_ENGINE_LIB
                )

    endif ()

	target_link_libraries(${PROJECT_NAME}
			${INTERNAL_EIGEN3_TARGET}
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

    # install (shared) lib to specified destinations
    InstallSharedLibrary(TARGET ${PROJECT_NAME})

    #import python dlls (if any, WIN32 only)
    include(cmake/ExportPythonDlls.cmake)
    export_python_dlls(${EROWCLOUDVIEWER_DEST_FOLDER})

endif ()

