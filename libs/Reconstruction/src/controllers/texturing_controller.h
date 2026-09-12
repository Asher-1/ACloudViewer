// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <atomic>
#include <filesystem>
#include <string>

#include "util/threading.h"

namespace colmap {

// Options for mesh texturing
struct TexturingOptions {
    // Show verbose information
    bool verbose = true;

    // Textured mesh file path (input)
    std::filesystem::path meshed_file_path;

    // Textured mesh output path
    std::filesystem::path textured_file_path;

    // COLMAP mesh texture mapping controls.
    double min_cos_normal_angle = 0.1;
    int min_visible_vertices = 3;
    int view_selection_smoothing_iterations = 3;
    int atlas_patch_padding = 2;
    int inpaint_radius = 5;
    bool apply_color_correction = true;
    double color_correction_regularization = 0.1;
    int num_threads = -1;
    double texture_scale_factor = 1.0;

    // Mesh source: "poisson", "delaunay", "advancing_front", or "auto".
    std::string mesh_source = "auto";

    // Check if options are valid
    bool Check() const;

    // Print the options to stdout
    void Print() const;
};

// Mesh texturing reconstruction controller backed by COLMAP texture mapping.
class TexturingReconstruction : public Thread {
public:
    TexturingReconstruction(const TexturingOptions& options,
                            const std::string& output_path);

    bool IsSuccess() const { return success_.load(); }

private:
    void Run();

    TexturingOptions options_;
    const std::string output_path_;
    std::atomic<bool> success_{false};
};

}  // namespace colmap
