# Build
file(GLOB_RECURSE ALL_HEADER_FILES "*.h")
set(SRC
    PointCloudIO.cpp
    file_format/FileXYZI.cpp
    file_format/FilePLY.cpp
)

# Create object library
add_library(tio OBJECT
			${ALL_HEADER_FILES}
            ${SRC}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})
ShowAndAbortOnWarning(tio)
cloudViewer_set_global_properties(tio)
cloudViewer_set_cloudViewer_lib_properties(tio)
cloudViewer_link_3rdparty_libraries(tio)
