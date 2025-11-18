// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <ecvMesh.h>

#include <memory>
#include <string>
#include <vector>

#include "base/reconstruction.h"
#include "util/alignment.h"
#include "util/misc.h"
#include "util/threading.h"

namespace colmap {
class Reconstruction;
namespace mvs {
class Workspace;
class DepthMap;
class NormalMap;
}  // namespace mvs
}  // namespace colmap

namespace cloudViewer {
namespace camera {
class PinholeCameraTrajectory;
}

struct TexturingOptions {
    // show verbose information
    bool verbose = true;

    // Display a 3D representation showing the a cloud and a list of camera with
    // their 6DOf poses
    bool show_cameras = false;

    // obj textured mesh save precision
    int save_precision = 5;

    // textured mesh file path
    std::string meshed_file_path = "";

    std::string textured_file_path = "";

    // Use depth maps and normal maps for visibility testing
    bool use_depth_normal_maps = true;

    // Depth map type: "photometric" or "geometric"
    std::string depth_map_type = "photometric";

    // Maximum depth error threshold for visibility check (relative)
    float max_depth_error = 0.01f;

    // Minimum normal consistency threshold (cosine of angle)
    // Lowered from 0.5f to 0.1f to allow more faces to pass quality check
    float min_normal_consistency = 0.1f;

    bool Check() const;
};

// Undistort images and export undistorted cameras, as required by the
// cloudViewer::DenseReconstructionWidget::Texturing class.
// Refactored to use depth maps and normal maps for improved texturing quality,
// following the mvs-texturing approach.
class TexturingReconstruction : public colmap::Thread {
public:
    TexturingReconstruction(const TexturingOptions& options,
                            const colmap::Reconstruction& reconstruction,
                            const std::string& image_path,
                            const std::string& output_path,
                            const std::vector<colmap::image_t>& image_ids =
                                    std::vector<colmap::image_t>());

private:
    void Run();

    bool Texturing(const colmap::image_t image_id, std::size_t index);

    // Get image index in workspace from image_id
    int GetWorkspaceImageIdx(const colmap::image_t image_id) const;

    // Check if a 3D point is visible in a view using depth map
    bool IsPointVisible(const Eigen::Vector3d& point3d,
                        const colmap::Image& image,
                        const colmap::Camera& camera,
                        int workspace_image_idx) const;

    // Compute view quality score using normal map
    float ComputeViewQuality(const Eigen::Vector3d& point3d,
                             const Eigen::Vector3d& face_normal,
                             const colmap::Image& image,
                             const colmap::Camera& camera,
                             int workspace_image_idx) const;

    // Filter camera trajectory based on depth/normal maps visibility
    // This creates a filtered trajectory with only cameras that have valid
    // depth/normal maps. Uses the provided mesh for visibility testing.
    std::shared_ptr<camera::PinholeCameraTrajectory> FilterCameraTrajectory(
            ccMesh* mesh) const;

    TexturingOptions options_;
    const std::string image_path_;
    const std::string output_path_;
    const std::vector<colmap::image_t> image_ids_;
    const colmap::Reconstruction& reconstruction_;
    std::unique_ptr<colmap::mvs::Workspace>
            workspace_;  // Optional workspace for depth/normal maps
    std::vector<std::string> image_names_;
    std::shared_ptr<camera::PinholeCameraTrajectory> camera_trajectory_;
};
}  // namespace cloudViewer
