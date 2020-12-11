file(GLOB_RECURSE ALL_HEADER_FILES "*.h")
set (SRC
    Cloud.cpp
    GridSubsampling.cpp
    contrib_nns.cpp
)

add_library(ml_contrib OBJECT ${ALL_HEADER_FILES} ${SRC})
cloudViewer_set_global_properties(ml_contrib)
cloudViewer_link_3rdparty_libraries(ml_contrib)
cloudViewer_set_cloudViewer_lib_properties(ml_contrib)
ShowAndAbortOnWarning(ml_contrib)
