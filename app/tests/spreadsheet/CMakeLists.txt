# Configured only by the parent GUI build with BUILD_UNIT_TESTS enabled.
# Links the production spreadsheet model/view; no MainWindow dependency.
if(NOT TARGET gtest)
    include(${CloudViewer_3RDPARTY_DIR}/googletest/googletest.cmake)
endif()

cloudviewer_add_plugin_test(test_spreadsheet_header
    SOURCES
        test_spreadsheet_header.cpp
        ../../ecvSpreadSheetView.cpp ../../ecvSpreadSheetView.h
    LIBS CV_DB_LIB CVCoreLib Qt::Widgets gtest
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/libs/VtkEngine
        ${CMAKE_SOURCE_DIR}/libs/CVPluginAPI/include
    DEFINES QT_USE_QSTRINGBUILDER)
set_target_properties(test_spreadsheet_header PROPERTIES AUTOMOC ON)
cloudviewer_add_test(NAME test_spreadsheet_header COMMAND test_spreadsheet_header)
cloudviewer_set_plugin_test_env_offscreen(test_spreadsheet_header)
set_tests_properties(test_spreadsheet_header PROPERTIES
    LABELS "spreadsheet;unit" TIMEOUT 60)