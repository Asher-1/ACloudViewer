#include "mvs/mesh_postprocessing.h"

#define TEST_NAME "mvs/mesh_postprocessing_test"
#include "util/testing.h"

#include <algorithm>


namespace colmap {
namespace mvs {

TEST(MeshPostProcessing, DefaultsEnabledAndValid) {
    MeshPostProcessingOptions options;
    EXPECT_TRUE(options.enabled);
    EXPECT_TRUE(options.Check());
}

TEST(MeshPostProcessing, RemovesDegenerateAndSmoothsInterior) {
    PlyMesh mesh;
    mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0.5f, 0.5f, 0.2f}};
    mesh.faces = {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}, {0, 0, 1}};
    MeshPostProcessingOptions options;
    options.remove_small_components = false;
    MeshPostProcessingStats stats;
    const PlyMesh out = PostProcessMesh(mesh, options, &stats);
    EXPECT_EQ(stats.removed_invalid_faces, 1u);
    EXPECT_EQ(out.faces.size(), 4u);
    bool found_center = false;
    float center_z = 0.0f;
    for (const PlyMeshVertex& vertex : out.vertices) {
        if (vertex.x == 0.5f && vertex.y == 0.5f) {
            found_center = true;
            center_z = vertex.z;
            break;
        }
    }
    EXPECT_TRUE(found_center);
    if (found_center) {
        EXPECT_NE(center_z, 0.2f);
    }
}

TEST(MeshPostProcessing, DisabledIsIdentity) {
    PlyMesh mesh;
    mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.faces = {{0, 1, 2}};
    MeshPostProcessingOptions options;
    options.enabled = false;
    EXPECT_EQ(PostProcessMesh(mesh, options).faces.size(), mesh.faces.size());
}

}  // namespace mvs
}  // namespace colmap
