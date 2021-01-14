set(FILE_IO_SRC
    PointCloudIO.cpp
    file_format/FileXYZI.cpp
    file_format/FilePLY.cpp
    )

set(SENSOR_IO_SRC
    sensor/RGBDVideoReader.cpp
    sensor/RGBDVideoMetadata.cpp
    )

set(IO_DEFINITIONS "")
if (BUILD_LIBREALSENSE)
    set(REALSENSE_SRC
        sensor/realsense/RSBagReader.cpp
        sensor/realsense/RealSenseSensor.cpp
        sensor/realsense/RealSenseSensorConfig.cpp
        )
    list(APPEND IO_DEFINITIONS BUILD_LIBREALSENSE)
endif ()

# Create object library
add_library(tio OBJECT
    ${FILE_IO_SRC}
    ${RPLY_SOURCE_FILES}
    ${LIBLZF_SOURCE_FILES}
    ${SENSOR_IO_SRC}
    ${REALSENSE_SRC}
    )
target_compile_definitions(tio PRIVATE ${IO_DEFINITIONS})
cloudViewer_show_and_abort_on_warning(tio)
cloudViewer_set_global_properties(tio)
cloudViewer_set_cloudViewer_lib_properties(tio)
cloudViewer_link_3rdparty_libraries(tio)
