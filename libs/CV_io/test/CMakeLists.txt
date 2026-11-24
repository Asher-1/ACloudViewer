# The SHP fixture suite uses the repository's GoogleTest dependency and the
# same Qt/production I/O libraries as its owner on both Qt 5 and Qt 6.
if(OPTION_USE_SHAPE_LIB)
    if(NOT TARGET gtest)
        include(${CloudViewer_3RDPARTY_DIR}/googletest/googletest.cmake)
    endif()
    cloudviewer_add_plugin_test(TestShpFilter
        SOURCES TestShpFilter.cpp TestShpFilter.h TestData.qrc
        LIBS CV_IO_LIB gtest Qt::Widgets)
    set_target_properties(TestShpFilter PROPERTIES AUTORCC ON)
    cloudviewer_add_test(NAME TestShpFilter COMMAND TestShpFilter
        LABELS "unit;io;shp")
    cloudviewer_set_plugin_test_env_offscreen(TestShpFilter)
endif()
