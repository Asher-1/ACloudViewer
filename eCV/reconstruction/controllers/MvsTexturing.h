// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvMesh.h>

#include <Eigen/Core>
#include <QImage>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "util/types.h"

namespace colmap {
class Image;
class Camera;
class Reconstruction;
namespace mvs {
class Workspace;
}
}  // namespace colmap

namespace cloudViewer {
namespace camera {
class PinholeCameraTrajectory;
}

// Internal structures for texturing
struct TextureView {
    std::size_t id;
    std::string image_file;
    int width;
    int height;
    Eigen::Matrix3f projection;
    Eigen::Matrix4f world_to_cam;
    Eigen::Vector3f pos;
    Eigen::Vector3f viewdir;

    // Image data (loaded on demand)
    mutable std::shared_ptr<QImage> image_data;

    TextureView(std::size_t id,
                const std::string& image_file,
                const Eigen::Matrix3f& projection,
                const Eigen::Matrix4f& world_to_cam,
                const Eigen::Vector3f& pos,
                const Eigen::Vector3f& viewdir,
                int width,
                int height);

    Eigen::Vector2f GetPixelCoords(const Eigen::Vector3f& vertex) const;
    bool ValidPixel(const Eigen::Vector2f& pixel) const;
    bool Inside(const Eigen::Vector3f& v1,
                const Eigen::Vector3f& v2,
                const Eigen::Vector3f& v3) const;
};

struct TexturePatch {
    int label;
    std::vector<size_t> faces;
    std::vector<Eigen::Vector2f> texcoords;
    std::shared_ptr<QImage> image;
    int min_x, min_y, max_x, max_y;

    TexturePatch(int label,
                 const std::vector<size_t>& faces,
                 const std::vector<Eigen::Vector2f>& texcoords,
                 std::shared_ptr<QImage> image,
                 int min_x,
                 int min_y,
                 int max_x,
                 int max_y);
};

struct ObjModel {
    struct Face {
        size_t vertex_ids[3];
        size_t texcoord_ids[3];
        size_t normal_ids[3];
    };

    struct Group {
        std::string material_name;
        std::vector<Face> faces;
    };

    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector2f> texcoords;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Group> groups;
    std::map<std::string, std::string>
            materials;  // material_name -> texture_file

    bool SaveToFiles(const std::string& prefix) const;
};

// Texture mapping implementation based on mvs-texturing approach
class MvsTexturing {
public:
    struct Options {
        bool verbose = true;
        float max_depth_error = 0.01f;
        float max_viewing_angle_deg = 75.0f;  // Maximum viewing angle in
                                              // degrees (mvs-texturing uses 75)
        bool use_depth_normal_maps =
                true;  // For visibility testing only, not quality
        bool use_gradient_magnitude =
                true;  // Use GMI (Gradient Magnitude Integral) for quality
    };

    MvsTexturing(const Options& options,
                 const colmap::Reconstruction& reconstruction,
                 colmap::mvs::Workspace* workspace,
                 const std::string& image_path);

    // Main texturing function
    // Uses the provided mesh object (must not be null)
    bool TextureMesh(ccMesh& mesh,
                     const camera::PinholeCameraTrajectory& camera_trajectory,
                     const std::string& output_path);

private:
    // Create texture views from camera trajectory
    void CreateTextureViews(
            const camera::PinholeCameraTrajectory& camera_trajectory);

    // Calculate data costs for each face-view combination
    void CalculateDataCosts(const ccMesh& mesh);

    // View selection using graph cut
    void SelectViews();

    // Generate texture patches
    void GenerateTexturePatches(const ccMesh& mesh);

    // Seam leveling (needs mesh to find shared vertices)
    void SeamLeveling(const ccMesh& mesh);

    // Generate texture atlases
    void GenerateTextureAtlases();

    // Build and save OBJ model
    bool SaveOBJModel(const std::string& output_path, const ccMesh& mesh);

    // Helper methods for visibility and quality testing
    int GetWorkspaceImageIdx(const colmap::image_t image_id) const;
    bool IsPointVisible(const Eigen::Vector3d& point3d,
                        const colmap::Image& image,
                        const colmap::Camera& camera,
                        int workspace_image_idx) const;

    Options options_;
    const colmap::Reconstruction& reconstruction_;
    colmap::mvs::Workspace* workspace_;
    const std::string image_path_;

    std::vector<std::unique_ptr<TextureView>> texture_views_;
    std::vector<std::unique_ptr<TexturePatch>> texture_patches_;
    std::unique_ptr<ObjModel> obj_model_;

    // Texture atlases: merged patches with updated texture coordinates
    struct TextureAtlas {
        unsigned int width;
        unsigned int height;
        std::shared_ptr<QImage> image;
        std::vector<size_t> face_ids;  // Mesh face indices in this atlas
        std::vector<Eigen::Vector2f>
                texcoords;  // Normalized texture coordinates [0,1]
        std::vector<size_t>
                texcoord_ids;  // Per-face texture coordinate indices
    };
    std::vector<TextureAtlas> texture_atlases_;

    // Mapping from view_id to image_id
    std::vector<colmap::image_t> view_to_image_id_;

    // Helper methods for graph operations
    void BuildAdjacencyGraph(const ccMesh& mesh);
    float ComputePairwiseCost(size_t face1,
                              size_t face2,
                              size_t view1,
                              size_t view2) const;

    // Data structures for texturing
    struct FaceProjectionInfo {
        size_t view_id;
        float quality;
        Eigen::Vector3f mean_color;
    };
    std::vector<std::vector<FaceProjectionInfo>>
            face_projection_infos_;    // [face_id] -> list of views
    std::vector<size_t> face_labels_;  // [face_id] -> selected view_id

    // Vertex projection info (following mvs-texturing)
    struct VertexProjectionInfo {
        size_t texture_patch_id;
        Eigen::Vector2f projection;
        std::vector<size_t> faces;

        bool operator<(const VertexProjectionInfo& other) const {
            return texture_patch_id < other.texture_patch_id;
        }
    };
    std::vector<std::vector<VertexProjectionInfo>>
            vertex_projection_infos_;  // [vertex_id] -> list of projections

    // Adjacency graph for mesh faces
    std::vector<std::vector<size_t>>
            face_adjacency_;  // [face_id] -> adjacent face_ids

    // Data costs for graph cut: [face_id][view_idx] -> cost
    std::vector<std::vector<std::pair<size_t, float>>>
            data_costs_;  // [face_id] -> [(view_id, cost)]

    // Seam edges: edges between faces with different labels
    struct SeamEdge {
        size_t v1, v2;        // vertex indices
        size_t face1, face2;  // face indices
    };
    std::vector<SeamEdge> seam_edges_;

    // Helper method to get subgraphs (connected components with same label)
    void GetSubgraphs(size_t label,
                      std::vector<std::vector<size_t>>* subgraphs) const;

    // Helper method to merge vertex projection infos
    void MergeVertexProjectionInfos();
};

}  // namespace cloudViewer
