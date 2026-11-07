# Vendored meshoptimizer subset used by TRELLIS and MVS mesh cleanup.
add_library(3rdparty_meshoptimizer STATIC
    ${CMAKE_CURRENT_LIST_DIR}/allocator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/indexgenerator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/simplifier.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vfetchoptimizer.cpp)
set_target_properties(3rdparty_meshoptimizer PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    FOLDER "3rdparty")
target_include_directories(3rdparty_meshoptimizer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
if (COMMAND cloudViewer_set_global_properties)
    cloudViewer_set_global_properties(3rdparty_meshoptimizer)
endif ()
