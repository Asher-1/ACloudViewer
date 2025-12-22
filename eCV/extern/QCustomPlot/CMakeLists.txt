cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})

# CC custom CMake script for QCustomPlot
# https://www.qcustomplot.com/

project( QCustomPlot )

add_library( ${PROJECT_NAME} STATIC )

target_sources( ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/qcustomplot.cpp
        ${CMAKE_CURRENT_LIST_DIR}/qcustomplot.h
)

target_include_directories( ${PROJECT_NAME}
	PUBLIC
	    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Qt5/Qt6 compatibility - use unified Qt:: prefix
target_link_libraries( ${PROJECT_NAME}
    Qt::Gui
)