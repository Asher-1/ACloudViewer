# Configured only by the parent GUI build with BUILD_UNIT_TESTS enabled.
# Links the production layout widgets and VTK engine; MainWindow is stubbed
# in the test source to the few symbols ecvMultiViewWidget.cpp references.
if(NOT TARGET gtest)
    include(${CloudViewer_3RDPARTY_DIR}/googletest/googletest.cmake)
endif()

if(USE_VTK_BACKEND AND TARGET QVTK_ENGINE_LIB)
    cloudviewer_add_plugin_test(test_layout_close
        SOURCES
            test_layout_close.cpp
            ../../ecvMultiViewWidget.cpp ../../ecvMultiViewWidget.h
            ../../ecvTabbedMultiViewWidget.cpp ../../ecvTabbedMultiViewWidget.h
            ../../ecvSpreadSheetView.cpp ../../ecvSpreadSheetView.h
            ../../ecvPythonView.cpp ../../ecvPythonView.h
            ../../ecvPythonCodeEditor.cpp ../../ecvPythonCodeEditor.h
            ../../ecvPythonSyntaxHighlighter.cpp ../../ecvPythonSyntaxHighlighter.h
        LIBS QVTK_ENGINE_LIB CV_DB_LIB CVCoreLib Qt::Widgets Qt::Network gtest
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}/../..
            ${CMAKE_SOURCE_DIR}/tests
            ${CMAKE_SOURCE_DIR}/libs/VtkEngine
            ${CMAKE_SOURCE_DIR}/libs/CVPluginAPI/include
        DEFINES QT_USE_QSTRINGBUILDER)
    set_target_properties(test_layout_close PROPERTIES AUTOMOC ON)
    # Showing production VTK views creates real OpenGL contexts; offscreen QPA
    # does not provide them. Keep this lifecycle regression in the graphics gate.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        find_program(LAYOUT_XVFB_RUN xvfb-run)
    endif()
    if(LAYOUT_XVFB_RUN)
        cloudviewer_add_test(NAME test_layout_close
            COMMAND ${LAYOUT_XVFB_RUN} -a $<TARGET_FILE:test_layout_close>)
        set_tests_properties(test_layout_close PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=xcb")
    else()
        cloudviewer_add_test(NAME test_layout_close COMMAND test_layout_close)
    endif()
    cloudviewer_set_plugin_test_env(test_layout_close)
    set_tests_properties(test_layout_close PROPERTIES
        LABELS "layout;integration;graphics" TIMEOUT 120)
endif()