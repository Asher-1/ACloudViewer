# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

set(FOLDER_NAME "scene")

# W9 (synthetic dataset generator): enabled with the W3-2a frame-aware data
# model. The remaining fork adaptations against upstream dbb41680 are marked
# inline in synthetic.cc and recorded in colmap_alignment_manifest.json
# (synthetic_dataset).
COLMAP_ADD_SOURCES(
    synthetic.h synthetic.cc
    pose_graph.h pose_graph.cc
    reconstruction_io_utils.h reconstruction_io_utils.cc
    reconstruction_io_text.h reconstruction_io_text.cc
    reconstruction_io_binary.h reconstruction_io_binary.cc
    reconstruction_io.h reconstruction_io.cc
    database_sqlite.cc
    reconstruction_clustering.h reconstruction_clustering.cc
    reconstruction_pruning.h reconstruction_pruning.cc
)

if (TESTS_ENABLED)
    COLMAP_ADD_TEST(synthetic_test synthetic_test.cc)
    COLMAP_ADD_TEST(pose_graph_test pose_graph_test.cc)
    COLMAP_ADD_TEST(reconstruction_io_test reconstruction_io_test.cc)
    COLMAP_ADD_TEST(reconstruction_clustering_test reconstruction_clustering_test.cc)
    COLMAP_ADD_TEST(reconstruction_pruning_test reconstruction_pruning_test.cc)
    COLMAP_ADD_TEST(reconstruction_matchers_test reconstruction_matchers_test.cc)
endif ()

# Moved from base/ (upstream scene/ parity).
COLMAP_ADD_SOURCES(
    camera.h camera.cc
    correspondence_graph.h correspondence_graph.cc
    database.h database.cc
    database_cache.h database_cache.cc
    frame.h frame.cc
    image.h image.cc
    point2d.h point2d.cc
    point3d.h point3d.cc
    projection.h projection.cc
    reconstruction.h reconstruction.cc
    reconstruction_manager.h reconstruction_manager.cc
    rig.h rig.cc
    scene_clustering.h scene_clustering.cc
    track.h track.cc
    two_view_geometry.h two_view_geometry.cc
    visibility_pyramid.h visibility_pyramid.cc
)
if (TESTS_ENABLED)
    COLMAP_ADD_TEST(camera_test camera_test.cc)
    COLMAP_ADD_TEST(correspondence_graph_test correspondence_graph_test.cc)
    COLMAP_ADD_TEST(database_cache_test database_cache_test.cc)
    COLMAP_ADD_TEST(database_test database_test.cc)
    COLMAP_ADD_TEST(frame_test frame_test.cc)
    COLMAP_ADD_TEST(image_test image_test.cc)
    COLMAP_ADD_TEST(point2d_test point2d_test.cc)
    COLMAP_ADD_TEST(point3d_test point3d_test.cc)
    COLMAP_ADD_TEST(projection_test projection_test.cc)
    COLMAP_ADD_TEST(reconstruction_test reconstruction_test.cc)
    COLMAP_ADD_TEST(reconstruction_manager_test reconstruction_manager_test.cc)
    COLMAP_ADD_TEST(rig_test rig_test.cc)
    COLMAP_ADD_TEST(scene_clustering_test scene_clustering_test.cc)
    COLMAP_ADD_TEST(track_test track_test.cc)
    COLMAP_ADD_TEST(visibility_pyramid_test visibility_pyramid_test.cc)
endif ()
