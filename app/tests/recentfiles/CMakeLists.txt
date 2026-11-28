# Configured only by the parent GUI build with BUILD_UNIT_TESTS enabled.
# Links the production recent-files menu logic; MainWindow is stubbed in the
# test source to the few symbols ecvRecentFiles.cpp references (same pattern
# as tests/layout).
if(NOT TARGET gtest)
    include(${CloudViewer_3RDPARTY_DIR}/googletest/googletest.cmake)
endif()

if(TARGET CVAppCommon AND TARGET gtest)
    cloudviewer_add_plugin_test(test_recent_files
        SOURCES test_recent_files.cpp
            ../../ecvRecentFiles.cpp ../../ecvRecentFiles.h
        LIBS CVAppCommon CV_DB_LIB CVCoreLib Qt::Widgets Qt::Network gtest
        INCLUDES ${CMAKE_SOURCE_DIR}/app
            ${CMAKE_SOURCE_DIR}/libs/VtkEngine
            ${CMAKE_SOURCE_DIR}/libs/CVPluginAPI/include
        DEFINES QT_USE_QSTRINGBUILDER)
    set_target_properties(test_recent_files PROPERTIES AUTOMOC ON)
    cloudviewer_add_test(NAME test_recent_files COMMAND test_recent_files
        LABELS recentfiles unit)
    cloudviewer_set_plugin_test_env_offscreen(test_recent_files)
    set_tests_properties(test_recent_files PROPERTIES TIMEOUT 30)
endif()