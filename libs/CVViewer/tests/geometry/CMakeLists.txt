target_sources(tests PRIVATE
    AccumulatedPoint.cpp
    EstimateNormals.cpp
    # HalfEdgeTriangleMesh.cpp  # TODO: Fix API
    # Image.cpp  # TODO: Fix Image API (PointerAs → PointerAt)
    # IntersectionTest.cpp  # TODO: Fix missing header
    # KDTreeFlann.cpp  # TODO: Fix PointCloud namespace
    # Line3D.cpp  # TODO: Fix Line3D API
    # LineSet.cpp  # TODO: Fix LineSet API (Clear → clear, PointCloud namespace)
    Octree.cpp
    # PointCloud.cpp  # TODO: Fix PointCloud API (colors_, points_)
    RGBDImage.cpp
    # TetraMesh.cpp  # TODO: Fix missing header
    # TriangleMesh.cpp  # TODO: Fix TriangleMesh API
    # VoxelGrid.cpp  # TODO: Fix visualization includes
)
