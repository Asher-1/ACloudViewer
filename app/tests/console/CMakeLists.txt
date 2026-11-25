# Exercise the production console without the VTK graphics/lifecycle runner.
if(TARGET CVAppCommon AND TARGET QVTK_ENGINE_LIB AND TARGET gtest)
    cloudviewer_add_plugin_test(test_console_errors
        SOURCES test_console_errors.cpp
            ../../ecvConsole.cpp ../../ecvConsole.h
        LIBS CVAppCommon QVTK_ENGINE_LIB Qt::Widgets Qt::Network gtest
        INCLUDES ${CMAKE_SOURCE_DIR}/app
            ${CMAKE_SOURCE_DIR}/libs/VtkEngine
            ${CMAKE_SOURCE_DIR}/libs/CVPluginAPI/include
        DEFINES QT_USE_QSTRINGBUILDER)
    set_target_properties(test_console_errors PROPERTIES AUTOMOC ON)
    cloudviewer_add_test(NAME test_console_errors COMMAND test_console_errors
        LABELS console unit)
    cloudviewer_set_plugin_test_env_offscreen(test_console_errors)
    set_tests_properties(test_console_errors PROPERTIES TIMEOUT 30)
endif()
