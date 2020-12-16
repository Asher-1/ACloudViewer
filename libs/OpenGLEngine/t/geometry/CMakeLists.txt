# Create object library
file(GLOB_RECURSE ALL_HEADER_FILES "*.h")
set(ALL_SOURCE_FILES
    PointCloud.cpp
    Image.cpp
    TensorListMap.cpp
    TriangleMesh.cpp
)

add_library(tgeometry OBJECT ${ALL_SOURCE_FILES} ${ALL_HEADER_FILES})
ShowAndAbortOnWarning(tgeometry)
cloudViewer_set_global_properties(tgeometry)
cloudViewer_set_cloudViewer_lib_properties(tgeometry)
cloudViewer_link_3rdparty_libraries(tgeometry)

