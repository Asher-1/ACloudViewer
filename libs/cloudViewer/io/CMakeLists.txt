# Build
file(GLOB IO_ALL_HEADER_FILES "*.h")
file(GLOB CLASS_IO_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE FILE_FORMAT_SOURCE_FILES "file_format/*.cpp")
set(IO_ALL_SOURCE_FILES ${CLASS_IO_SOURCE_FILES} ${FILE_FORMAT_SOURCE_FILES})

file(GLOB      RPC_HEADER_FILES "rpc/*.h")
file(GLOB      RPC_SOURCE_FILES "rpc/*.cpp")
set(IO_ALL_HEADER_FILES ${IO_ALL_HEADER_FILES} ${RPC_HEADER_FILES})
set(IO_ALL_SOURCE_FILES ${IO_ALL_SOURCE_FILES} ${RPC_SOURCE_FILES})

if (BUILD_AZURE_KINECT)
    file(GLOB_RECURSE SENSOR_HEADER_FILES "sensor/*.h")
    file(GLOB_RECURSE SENSOR_SOURCE_FILES "sensor/*.cpp")
    set(IO_ALL_HEADER_FILES ${IO_ALL_HEADER_FILES} ${SENSOR_HEADER_FILES})
    set(IO_ALL_SOURCE_FILES ${IO_ALL_SOURCE_FILES} ${SENSOR_SOURCE_FILES})
endif ()

# Create object library
add_library(io OBJECT
            ${IO_ALL_HEADER_FILES}
            ${IO_ALL_SOURCE_FILES}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})

cloudViewer_show_and_abort_on_warning(io)
cloudViewer_set_global_properties(io)
cloudViewer_set_cloudViewer_lib_properties(io)
cloudViewer_link_3rdparty_libraries(io)
