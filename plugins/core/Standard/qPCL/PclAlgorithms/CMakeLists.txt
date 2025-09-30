cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

set_property(GLOBAL PROPERTY USE_FOLDERS On)

if (NOT MSVC AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -lpthread")
endif ()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${ACloudViewerPlugins_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${EXTERNAL_LIBS_INCLUDE_DIR})

project(QPCL_PLUGIN_ALGORIGHM_LIB)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

if (USE_PCL_BACKEND)
    include_directories(${QPCL_ENGINE_LIB_SOURCE_DIR})

    # CURRENT DIR
    file(GLOB header_list *.h)
    file(GLOB source_list *.cpp)
    file(GLOB_RECURSE ui_list ${CMAKE_CURRENT_SOURCE_DIR}/*.ui)
    qt5_wrap_ui(ui_list_all ${ui_list})

    # PCL FILTERS
    file(GLOB_RECURSE filter_list ${CMAKE_CURRENT_SOURCE_DIR}/Filters/*)
    source_group("PclAlgorithms\\Filters" FILES ${filter_list})

    # PCL FEATURES
    file(GLOB_RECURSE feature_list ${CMAKE_CURRENT_SOURCE_DIR}/Features/*)
    source_group("PclAlgorithms\\Features" FILES ${feature_list})

    # PCL SURFACES
    file(GLOB_RECURSE surface_list ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/*)
    source_group("PclAlgorithms\\Surfaces" FILES ${surface_list})

    # PCL SEGMENTATIONS
    file(GLOB_RECURSE segment_list ${CMAKE_CURRENT_SOURCE_DIR}/Segmentations/*)
    source_group("PclAlgorithms\\Segmentations" FILES ${segment_list})

    # PCL RECOGNITIONS
    file(GLOB_RECURSE recognition_list ${CMAKE_CURRENT_SOURCE_DIR}/Recognitions/*)
    source_group("PclAlgorithms\\Recognitions" FILES ${recognition_list})

    # add library
    add_library(${PROJECT_NAME}
            ${header_list}
            ${source_list}
            ${ui_list_all}
            ${filter_list}
            ${feature_list}
            ${surface_list}
            ${segment_list}
            ${recognition_list})

    target_include_directories(${PROJECT_NAME}
            PUBLIC
            ${PCL_INCLUDE_DIRS}
            )
    if (CMAKE_VERSION VERSION_LESS 3.13)
        link_directories(${PCL_LIBRARY_DIRS})
    else ()
        target_link_directories(${PROJECT_NAME}
                PUBLIC
                ${PCL_LIBRARY_DIRS}
                )
    endif ()
    target_compile_definitions(${PROJECT_NAME} PRIVATE ${PCL_DEFINITIONS})
    target_link_libraries(${PROJECT_NAME}
            CVPluginAPI
            QPCL_ENGINE_LIB)
    target_link_libraries(${PROJECT_NAME} 3rdparty_pcl)
endif ()
