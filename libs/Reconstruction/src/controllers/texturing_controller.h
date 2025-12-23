// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "base/reconstruction.h"
#include "mvs/workspace.h"
#include "util/alignment.h"
#include "util/misc.h"
#include "util/threading.h"

// Forward declarations for CloudViewer types
class ccMesh;

namespace cloudViewer {
namespace camera {
class PinholeCameraTrajectory;
}
}  // namespace cloudViewer

namespace colmap {

class Reconstruction;

// Options for mesh texturing
struct TexturingOptions {
    // Show verbose information
    bool verbose = true;

    // Textured mesh file path (input)
    std::string meshed_file_path = "";

    // Textured mesh output path
    std::string textured_file_path = "";

    // Use depth maps and normal maps for visibility testing
    bool use_depth_normal_maps = true;

    // Depth map type: "photometric" or "geometric"
    std::string depth_map_type = "geometric";

    // Maximum depth error threshold for visibility check (relative)
    double max_depth_error = 0.01;

    // Minimum normal consistency threshold (cosine of angle)
    double min_normal_consistency = 0.1;

    // Maximum viewing angle in degrees for texture view selection
    double max_viewing_angle_deg = 75.0;

    // Use gradient magnitude image (GMI) for texture quality
    bool use_gradient_magnitude = false;

    // Mesh source: "poisson", "delaunay", or "auto"
    std::string mesh_source = "auto";

    // Check if options are valid
    bool Check() const;

    // Print the options to stdout
    void Print() const;
};

// Mesh texturing reconstruction controller
// Uses MVS depth/normal maps for improved texturing quality
class TexturingReconstruction : public Thread {
public:
    TexturingReconstruction(
            const TexturingOptions& options,
            const Reconstruction& reconstruction,
            const std::string& image_path,
            const std::string& output_path,
            const std::vector<image_t>& image_ids = std::vector<image_t>());

private:
    void Run();

    bool Texturing(const image_t image_id, std::size_t index);

    // Get image index in workspace from image_id
    int GetWorkspaceImageIdx(const image_t image_id) const;

    // Check if a 3D point is visible in a view using depth map
    bool IsPointVisible(const Eigen::Vector3d& point3d,
                        const Image& image,
                        const Camera& camera,
                        int workspace_image_idx) const;

    // Compute view quality score using normal map
    float ComputeViewQuality(const Eigen::Vector3d& point3d,
                             const Eigen::Vector3d& face_normal,
                             const Image& image,
                             const Camera& camera,
                             int workspace_image_idx) const;

    // Filter camera trajectory based on depth/normal maps visibility
    std::shared_ptr<cloudViewer::camera::PinholeCameraTrajectory>
    FilterCameraTrajectory(ccMesh* mesh) const;

    TexturingOptions options_;
    const std::string image_path_;
    const std::string output_path_;
    const std::vector<image_t> image_ids_;
    const Reconstruction& reconstruction_;
    std::unique_ptr<mvs::Workspace> workspace_;
    std::vector<std::string> image_names_;
    std::shared_ptr<cloudViewer::camera::PinholeCameraTrajectory>
            camera_trajectory_;
};

}  // namespace colmap
