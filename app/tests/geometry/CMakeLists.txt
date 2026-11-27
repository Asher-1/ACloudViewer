# Configured only by the parent GUI build with BUILD_UNIT_TESTS enabled.
# Pure 2D geometry logic from the app layer; no GUI or VTK dependency.
if(NOT TARGET gtest)
    include(${CloudViewer_3RDPARTY_DIR}/googletest/googletest.cmake)
endif()

if(TARGET CV_DB_LIB AND TARGET CVCoreLib AND TARGET gtest)
    cloudviewer_add_plugin_test(test_inner_rect
        SOURCES test_inner_rect.cpp
            ../../ecvInnerRect2DFinder.cpp ../../ecvInnerRect2DFinder.h
        LIBS CV_DB_LIB CVCoreLib gtest
        INCLUDES ${CMAKE_SOURCE_DIR}/app)
    cloudviewer_add_test(NAME test_inner_rect COMMAND test_inner_rect
        LABELS geometry unit)
    set_tests_properties(test_inner_rect PROPERTIES TIMEOUT 30)
endif()