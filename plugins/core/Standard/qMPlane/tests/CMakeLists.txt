if(NOT TARGET 3rdparty_googletest OR NOT COMMAND cloudviewer_add_test)
    message(FATAL_ERROR "qMPlane tests require the root testing configuration; enable BUILD_UNIT_TESTS")
endif()

cloudviewer_add_plugin_test(QMPLANE_PLUGIN_TESTS
    SOURCES
        ../src/ccItemTree.cpp
        item_tree_test.cpp
        mocks/ccMainAppInterfaceMock.h
    INCLUDES ../src
    LIBS
        CVCoreLib CV_DB_LIB CV_IO_LIB CVPluginAPI
        3rdparty_googletest gtest_main
)

cloudviewer_add_test(NAME qMPlane.item_tree COMMAND QMPLANE_PLUGIN_TESTS)
cloudviewer_set_plugin_test_env(qMPlane.item_tree)
