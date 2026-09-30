# Regression gate for the composite aggregation data layer (massive-actor
# acceleration). Pure data-structure tests — no renderer/window required.
# Gated behind VTKENGINE_BUILD_TESTS so CI stays unaffected by default.

enable_testing()
if (NOT TARGET gtest_main)
    include(${CloudViewer_3RDPARTY_DIR}/googletest/googletest.cmake)
endif ()

add_executable(test_composite_block_registry test_composite_block_registry.cpp)
# Qt was built with -reduce-relocations: every target touching Qt headers
# must compile with -fPIC (-fPIE is NOT enough, per qglobal.h).
set_target_properties(test_composite_block_registry PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(test_composite_block_registry PRIVATE -fPIC)

target_link_libraries(test_composite_block_registry PRIVATE
    QVTK_ENGINE_LIB
    3rdparty_vtk
    gtest_main)

target_include_directories(test_composite_block_registry PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../VtkRendering/Core)

add_test(NAME composite_block_registry
         COMMAND test_composite_block_registry)
cloudviewer_set_bin_runtime_output_directory(test_composite_block_registry)
cloudviewer_set_plugin_test_env(composite_block_registry)
