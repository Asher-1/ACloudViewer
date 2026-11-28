# Configured only by the parent GUI build with BUILD_UNIT_TESTS enabled.
# Reuse the selected Qt and production libraries; do not compile private copies
# of the shortcut registry or infer flags from generator-specific build files.
if(NOT TARGET gtest)
    include(${CloudViewer_3RDPARTY_DIR}/googletest/googletest.cmake)
endif()

cloudviewer_add_plugin_test(test_shortcuts
    SOURCES
        test_shortcuts.cpp
        ../../ecvShortcutDialog.cpp ../../ecvShortcutDialog.h
        ../../ecvPythonCodeEditor.cpp ../../ecvPythonCodeEditor.h
        ../../ecvPythonView.cpp ../../ecvPythonView.h
        ../../ecvPythonSyntaxHighlighter.cpp ../../ecvPythonSyntaxHighlighter.h
    LIBS CV_DB_LIB CVCoreLib Qt::Widgets gtest
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/libs/VtkEngine
    DEFINES QT_USE_QSTRINGBUILDER SHORTCUT_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")
set_target_properties(test_shortcuts PROPERTIES
    AUTOMOC ON AUTOUIC ON
    AUTOUIC_SEARCH_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../ui_templates")
cloudviewer_add_test(NAME test_shortcuts COMMAND test_shortcuts)
cloudviewer_set_plugin_test_env_offscreen(test_shortcuts)
set_tests_properties(test_shortcuts PROPERTIES LABELS "shortcuts;unit" TIMEOUT 60)

if(USE_VTK_BACKEND AND TARGET QVTK_ENGINE_LIB)
    cloudviewer_add_plugin_test(test_vtk_shortcuts
        SOURCES test_vtk_shortcuts.cpp
        LIBS QVTK_ENGINE_LIB CV_DB_LIB Qt::Widgets gtest
        INCLUDES ${CMAKE_SOURCE_DIR}/libs/VtkEngine ${CMAKE_SOURCE_DIR}/tests)
    # Enabling the orientation widget calls Render(); keep that observer
    # regression in the explicit graphics row even with offscreen Qt available.
    cloudviewer_add_test(NAME test_vtk_shortcuts COMMAND test_vtk_shortcuts
        --gtest_filter=-VtkShortcuts.OrientationGizmoObserverPreservesZoom)
    cloudviewer_set_plugin_test_env_offscreen(test_vtk_shortcuts)
    set_tests_properties(test_vtk_shortcuts PROPERTIES
        LABELS "shortcuts;integration" TIMEOUT 60)

    # Actual OpenGL key events are a separate, explicit graphics gate. The
    # default root CI profile selects only the non-graphics registrations.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        find_program(SHORTCUT_XVFB_RUN xvfb-run)
        if(SHORTCUT_XVFB_RUN)
            cloudviewer_add_test(NAME test_vtk_shortcuts_graphics
                COMMAND ${SHORTCUT_XVFB_RUN} -a $<TARGET_FILE:test_vtk_shortcuts>)
            cloudviewer_set_plugin_test_env(test_vtk_shortcuts_graphics)
            set_tests_properties(test_vtk_shortcuts_graphics PROPERTIES
                LABELS "shortcuts;integration;graphics" TIMEOUT 120
                ENVIRONMENT "QT_QPA_PLATFORM=xcb;ACV_SHORTCUT_GRAPHICS_TEST=1")
        endif()
    endif()
endif()
