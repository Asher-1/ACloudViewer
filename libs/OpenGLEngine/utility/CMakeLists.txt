# build
file(GLOB ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(utility OBJECT ${ALL_SOURCE_FILES})
cloudViewer_show_and_abort_on_warning(utility)
cloudViewer_set_global_properties(utility)
cloudViewer_set_cloudViewer_lib_properties(utility)
cloudViewer_link_3rdparty_libraries(utility)
