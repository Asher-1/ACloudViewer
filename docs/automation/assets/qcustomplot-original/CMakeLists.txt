cmake_minimum_required(VERSION 3.16)
project(QCustomPlotOriginalIllustrations LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC OFF)
set(CMAKE_AUTORCC OFF)

find_package(Qt5 5.12 REQUIRED COMPONENTS Core Gui Widgets PrintSupport)

set(ACV_SOURCE_ROOT "" CACHE PATH "ACloudViewer source root containing libs/VtkEngine and core/include")
if(NOT EXISTS "${ACV_SOURCE_ROOT}/libs/VtkEngine/Tools/Common/qcustomplot.cpp" OR
   NOT EXISTS "${ACV_SOURCE_ROOT}/core/include/QtCompat.h")
    message(FATAL_ERROR "Set ACV_SOURCE_ROOT to the ACloudViewer source tree")
endif()
add_executable(qcp_original_illustrations
    main.cpp
    "${ACV_SOURCE_ROOT}/libs/VtkEngine/Tools/Common/qcustomplot.cpp"
    "${ACV_SOURCE_ROOT}/libs/VtkEngine/Tools/Common/qcustomplot.h")
target_include_directories(qcp_original_illustrations PRIVATE
    "${ACV_SOURCE_ROOT}/libs/VtkEngine/Tools/Common"
    "${ACV_SOURCE_ROOT}/core/include")
target_link_libraries(qcp_original_illustrations PRIVATE
    Qt5::Core Qt5::Gui Qt5::Widgets Qt5::PrintSupport)

