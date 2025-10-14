// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <memory>

#include "base/reconstruction.h"
#include "util/alignment.h"
#include "util/misc.h"
#include "util/threading.h"

namespace colmap {
class Reconstruction;
}
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

    bool Check() const;
};

// Undistort images and export undistorted cameras, as required by the
// cloudViewer::DenseReconstructionWidget::Texturing class.
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

    TexturingOptions options_;
    const std::string image_path_;
    const std::string output_path_;
    const std::vector<colmap::image_t> image_ids_;
    const colmap::Reconstruction& reconstruction_;
    std::vector<std::string> image_names_;
    std::shared_ptr<camera::PinholeCameraTrajectory> camera_trajectory_;
};
}  // namespace cloudViewer
