file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE ALL_HEADER_FILES "*.h")

add_library(pipelines OBJECT ${ALL_SOURCE_FILES} ${ALL_HEADER_FILES})
cloudViewer_show_and_abort_on_warning(pipelines)
cloudViewer_set_global_properties(pipelines)
cloudViewer_set_cloudViewer_lib_properties(pipelines)
cloudViewer_link_3rdparty_libraries(pipelines)
