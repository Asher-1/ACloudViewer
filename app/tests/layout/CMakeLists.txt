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

    # Component tests above intentionally stub MainWindow. Exercise real factory
    # wiring and the application selection coordinator in a separate binary.
    get_target_property(view_linkage_sources ${MAIN_APP_NAME} SOURCES)
    list(FILTER view_linkage_sources EXCLUDE REGEX "(^|/)main\\.cpp$")
    cloudviewer_add_plugin_test(test_view_linkage
        SOURCES test_view_linkage.cpp ${view_linkage_sources}
        LIBS "$<TARGET_PROPERTY:${MAIN_APP_NAME},LINK_LIBRARIES>" gtest
        INCLUDES "$<TARGET_PROPERTY:${MAIN_APP_NAME},INCLUDE_DIRECTORIES>"
        DEFINES "$<TARGET_PROPERTY:${MAIN_APP_NAME},COMPILE_DEFINITIONS>")
    set_target_properties(test_view_linkage PROPERTIES AUTOMOC ON)
    target_compile_options(test_view_linkage PRIVATE
        "$<TARGET_PROPERTY:${MAIN_APP_NAME},COMPILE_OPTIONS>")
    target_link_options(test_view_linkage PRIVATE
        "$<TARGET_PROPERTY:${MAIN_APP_NAME},LINK_OPTIONS>")
    target_link_directories(test_view_linkage PRIVATE
        "$<TARGET_PROPERTY:${MAIN_APP_NAME},LINK_DIRECTORIES>")
    # The production target owns generated UI/resources shared by the binaries.
    add_dependencies(test_view_linkage ${MAIN_APP_NAME})
    if(LAYOUT_XVFB_RUN)
        cloudviewer_add_test(NAME test_view_linkage
            COMMAND ${LAYOUT_XVFB_RUN} -a $<TARGET_FILE:test_view_linkage>)
        set_tests_properties(test_view_linkage PROPERTIES
            ENVIRONMENT "QT_QPA_PLATFORM=xcb")
    else()
        cloudviewer_add_test(NAME test_view_linkage COMMAND test_view_linkage)
    endif()
    cloudviewer_set_plugin_test_env(test_view_linkage)
    set_tests_properties(test_view_linkage PROPERTIES
        LABELS "layout;selection;integration;graphics" TIMEOUT 240)
endif()