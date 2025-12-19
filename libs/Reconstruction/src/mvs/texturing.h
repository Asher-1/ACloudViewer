// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <QImage>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "util/types.h"

// Forward declarations for CloudViewer types
class ccMesh;

namespace cloudViewer {
namespace camera {
class PinholeCameraTrajectory;
}
}  // namespace cloudViewer

namespace colmap {
class Image;
class Camera;
class Reconstruction;

namespace mvs {
class Workspace;

// Camera view for texturing
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

    // Project 3D point to 2D pixel coordinates
    Eigen::Vector2f GetPixelCoords(const Eigen::Vector3f& vertex) const;

    // Check if pixel is within image bounds
    bool ValidPixel(const Eigen::Vector2f& pixel) const;

    // Check if triangle is inside view frustum
    bool Inside(const Eigen::Vector3f& v1,
                const Eigen::Vector3f& v2,
                const Eigen::Vector3f& v3) const;
};

// Texture patch (atlas region)
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

// MVS-based texture mapping implementation
// Based on the mvs-texturing approach with depth/normal map guidance
class MvsTexturing {
public:
    struct Options {
        bool verbose = true;
        float max_depth_error = 0.01f;
        float max_viewing_angle_deg =
                75.0f;                       // Maximum viewing angle in degrees
        bool use_depth_normal_maps = true;   // For visibility testing
        bool use_gradient_magnitude = true;  // Use GMI for quality
    };

    MvsTexturing(const Options& options,
                 const Reconstruction& reconstruction,
                 Workspace* workspace,
                 const std::string& image_path);

    // Main texturing function
    // Uses ccMesh from CloudViewer
    bool TextureMesh(ccMesh& mesh,
                     const ::cloudViewer::camera::PinholeCameraTrajectory&
                             camera_trajectory,
                     const std::string& output_path);

private:
    // Create texture views from camera trajectory
    void CreateTextureViews(
            const ::cloudViewer::camera::PinholeCameraTrajectory&
                    camera_trajectory);

    // Calculate data costs for each face-view combination
    void CalculateDataCosts(const ccMesh& mesh);

    // View selection using graph cut
    void SelectViews();

    // Generate texture patches
    void GenerateTexturePatches(const ccMesh& mesh);

    // Seam leveling
    void SeamLeveling(const ccMesh& mesh);

    // Generate texture atlases
    void GenerateTextureAtlases();

    // Build and save OBJ model
    bool SaveOBJModel(const std::string& output_path, const ccMesh& mesh);

    // Helper methods for visibility and quality testing
    int GetWorkspaceImageIdx(const image_t image_id) const;
    bool IsPointVisible(const Eigen::Vector3d& point3d,
                        const colmap::Image& image,
                        const colmap::Camera& camera,
                        int workspace_image_idx) const;

    // GMI (Gradient Magnitude Image) methods
    QImage ComputeGradientMagnitudeImage(const QImage& image);
    float CalculateGMIQuality(size_t view_id,
                              const Eigen::Vector2f& p1,
                              const Eigen::Vector2f& p2,
                              const Eigen::Vector2f& p3);
    bool IsPointInTriangle(float px,
                           float py,
                           const Eigen::Vector2f& p1,
                           const Eigen::Vector2f& p2,
                           const Eigen::Vector2f& p3) const;

    Options options_;
    const Reconstruction& reconstruction_;
    Workspace* workspace_;
    const std::string image_path_;

    std::vector<std::unique_ptr<TextureView>> texture_views_;
    std::vector<std::unique_ptr<TexturePatch>> texture_patches_;

    // Texture atlases
    struct TextureAtlas {
        unsigned int width;
        unsigned int height;
        std::shared_ptr<QImage> image;
        std::vector<size_t> face_ids;
        std::vector<Eigen::Vector2f> texcoords;
        std::vector<size_t> texcoord_ids;
    };
    std::vector<TextureAtlas> texture_atlases_;

    // Mapping from view_id to image_id
    std::vector<image_t> view_to_image_id_;

    // GMI (Gradient Magnitude Images) for each texture view
    mutable std::vector<QImage> gradient_magnitude_images_;

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
    std::vector<std::vector<FaceProjectionInfo>> face_projection_infos_;
    std::vector<size_t> face_labels_;

    // Vertex projection info
    struct VertexProjectionInfo {
        size_t texture_patch_id;
        Eigen::Vector2f projection;
        std::vector<size_t> faces;

        bool operator<(const VertexProjectionInfo& other) const {
            return texture_patch_id < other.texture_patch_id;
        }
    };
    std::vector<std::vector<VertexProjectionInfo>> vertex_projection_infos_;

    // Adjacency graph for mesh faces
    std::vector<std::vector<size_t>> face_adjacency_;

    // Data costs for graph cut
    std::vector<std::vector<std::pair<size_t, float>>> data_costs_;

    // Seam edges
    struct SeamEdge {
        size_t v1, v2;
        size_t face1, face2;
    };
    std::vector<SeamEdge> seam_edges_;

    // Helper methods
    void GetSubgraphs(size_t label,
                      std::vector<std::vector<size_t>>* subgraphs) const;
    void MergeVertexProjectionInfos();
};

}  // namespace mvs
}  // namespace colmap
