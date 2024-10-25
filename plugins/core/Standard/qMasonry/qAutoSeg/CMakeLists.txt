cmake_minimum_required(VERSION 3.18)

# ACloudViewer Masonry Manual Segmentation plugin
option(PLUGIN_STANDARD_MASONRY_QAUTO_SEG "Check to install QAutoSeg plugin" OFF)

if (PLUGIN_STANDARD_MASONRY_QAUTO_SEG)

    project(QAUTO_SEG_PLUGIN)

    AddPlugin(NAME ${PROJECT_NAME})

    include_directories(${QPCL_ENGINE_LIB_SOURCE_DIR})

    target_sources(${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/qAutoSeg.h
            ${CMAKE_CURRENT_LIST_DIR}/qAutoSeg.cpp
            ${CMAKE_CURRENT_LIST_DIR}/profileImportDlg.h
            ${CMAKE_CURRENT_LIST_DIR}/profileImportDlg.cpp
            ${CMAKE_CURRENT_LIST_DIR}/profileImportDlg.ui
            )

    target_include_directories(${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            )

    if (USE_PCL_BACKEND)
        # [C/C++]>[General]>[Additional Include Directories]
        target_include_directories(${PROJECT_NAME}
                PUBLIC
                ${PCL_INCLUDE_DIRS}
                )
        # For Use Not PreCompiled Features
        #add_definitions( -DPCL_NO_PRECOMPILE )
        if (CMAKE_VERSION VERSION_LESS 3.13)
            link_directories(${PCL_LIBRARY_DIRS})
        else ()
            target_link_directories(${PROJECT_NAME}
                    PUBLIC
                    ${PCL_LIBRARY_DIRS}
                    )
        endif ()
        target_compile_definitions(${PROJECT_NAME} PUBLIC ${PCL_DEFINITIONS})
        target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} QPCL_ENGINE_LIB)
    endif ()

    if (BUILD_OPENCV)
        cloudViewer_link_static_lib(${PROJECT_NAME} 3rdparty_opencv)
        target_compile_definitions(${PROJECT_NAME} PRIVATE CV_STATIC_LIB)
        # target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
        # target_link_libraries(${PROJECT_NAME} 3rdparty_opencv)
        # target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    else ()
        message("Please turn BUILD_OPENCV to ON and try again...")
        set(BUILD_OPENCV ON)
    endif ()
endif ()
