target_sources(tests PRIVATE
    FeatureIO.cpp
    IJsonConvertibleIO.cpp
    ImageIO.cpp
    # OctreeIO.cpp  # TODO: Fix API
    PinholeCameraTrajectoryIO.cpp
    # PointCloudIO.cpp  # TODO: Fix PointCloud namespace
    PoseGraphIO.cpp
    TriangleMeshIO.cpp
    # VoxelGridIO.cpp  # TODO: Fix missing header
)
