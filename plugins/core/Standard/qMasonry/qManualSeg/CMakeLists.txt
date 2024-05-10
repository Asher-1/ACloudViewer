cmake_minimum_required(VERSION 3.18)

# ErowCloudViewer Masonry Manual Segmentation plugin
option(PLUGIN_STANDARD_MASONRY_QMANUAL_SEG "Check to install qManualSeg plugin" OFF)

if (PLUGIN_STANDARD_MASONRY_QMANUAL_SEG)

    project(QMANUAL_SEG_PLUGIN)

    AddPlugin(NAME ${PROJECT_NAME})

    include_directories(${QPCL_ENGINE_LIB_SOURCE_DIR})

    target_sources(${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/qManualSeg.h
            ${CMAKE_CURRENT_LIST_DIR}/qManualSeg.cpp
            ${CMAKE_CURRENT_LIST_DIR}/profileImportDlg.h
            ${CMAKE_CURRENT_LIST_DIR}/profileImportDlg.cpp
            ${CMAKE_CURRENT_LIST_DIR}/profileImportDlg.ui
            )

    target_include_directories(${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            )

    if (BUILD_OPENCV)
        target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${PROJECT_NAME} 3rdparty_opencv)
    else ()
        message("Please turn BUILD_OPENCV to ON and try again...")
        set(BUILD_OPENCV ON)
    endif ()

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

    #    # Find PCL
    #    find_package( PCL 1.9.1 REQUIRED )
    #
    #    # Find OpenCV
    #    #set( OpenCV_DIR "C:/opencv/build" ) #DGM: can't do that as it will obviously conflict with most of the users settings ;)
    #    find_package( OpenCV REQUIRED )
    #
    #    if( PCL_FOUND AND OpenCV_FOUND )
    #      # [C/C++]>[General]>[Additional Include Directories]
    #      include_directories( ${PCL_INCLUDE_DIRS} )
    #      include_directories( ${OpenCV_INCLUDE_DIRS} )
    #
    #      # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
    #      target_compile_definitions(${PROJECT_NAME} PRIVATE ${PCL_DEFINITIONS} )
    #
    #      # For Use Not PreCompiled Features
    #      #add_definitions( -DPCL_NO_PRECOMPILE )
    #
    #      # [Linker]>[General]>[Additional Library Directories]
    #      link_directories( ${PCL_LIBRARY_DIRS} )
    #      link_directories( ${OpenCV_LIB_DIR} )
    #
    #      # [Linker]>[Input]>[Additional Dependencies]
    #      target_link_libraries( QMANUAL_SEG_PLUGIN ${PCL_LIBRARIES} )
    #      target_link_libraries( QMANUAL_SEG_PLUGIN ${OpenCV_LIBS} )
    #
    #      #import OpenCV dlls (if any, WIN32 only)
    #      include( ../cmake/ExportOpenCVDlls.cmake )
    #      export_OpenCV_dlls( ${EROWCLOUDVIEWER_DEST_FOLDER} )
    #    endif()

endif ()
