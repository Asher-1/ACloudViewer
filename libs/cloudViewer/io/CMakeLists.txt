# Build
add_library(io OBJECT file_format/FileASSIMP.cpp)

target_sources(io PRIVATE
    FileFormatIO.cpp
    FileIOFactory.cpp
    ImageWarpingFieldIO.cpp
    ModelIO.cpp
    PointCloudIO.cpp
    PoseGraphIO.cpp
    TriangleMeshIO.cpp
)

target_sources(io PRIVATE
    rpc/BufferConnection.cpp
    rpc/Connection.cpp
    rpc/DummyReceiver.cpp
    rpc/MessageProcessorBase.cpp
    rpc/MessageUtils.cpp
    rpc/RemoteFunctions.cpp
    rpc/ZMQContext.cpp
    rpc/ZMQReceiver.cpp
)

if (BUILD_AZURE_KINECT)
    target_sources(io PRIVATE
        sensor/azure_kinect/AzureKinectRecorder.cpp
        sensor/azure_kinect/AzureKinectSensor.cpp
        sensor/azure_kinect/AzureKinectSensorConfig.cpp
        sensor/azure_kinect/K4aPlugin.cpp
        sensor/azure_kinect/MKVMetadata.cpp
        sensor/azure_kinect/MKVReader.cpp
        sensor/azure_kinect/MKVWriter.cpp
    )
endif ()

cloudViewer_show_and_abort_on_warning(io)
cloudViewer_set_global_properties(io)
cloudViewer_set_cloudViewer_lib_properties(io)
cloudViewer_link_3rdparty_libraries(io)
