// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <string>

#include "util/ply.h"

namespace colmap {
namespace mvs {

// Shared post-meshing cleanup. The default keeps the mesher's face density,
// removes isolated/invalid geometry, and applies a short boundary-preserving
// Taubin pass so texture charts are not dominated by sliver noise.
struct MeshPostProcessingOptions {
    bool enabled = true;
    bool remove_small_components = true;
    bool remove_degenerate_faces = true;
    bool simplify = false;
    bool smooth = true;
    bool preserve_boundary = true;
    double prune_error = 0.02;
    double target_face_ratio = 1.0;
    double simplify_error = 0.01;
    double max_aspect_ratio = 100.0;
    int smoothing_iterations = 3;
    double smoothing_lambda = 0.5;
    double smoothing_mu = -0.53;

    bool Check() const;
};

struct MeshPostProcessingStats {
    size_t input_vertices = 0;
    size_t input_faces = 0;
    size_t output_vertices = 0;
    size_t output_faces = 0;
    size_t removed_invalid_faces = 0;
    size_t removed_small_component_faces = 0;
    size_t simplified_faces = 0;
};

PlyMesh PostProcessMesh(const PlyMesh& mesh,
                        const MeshPostProcessingOptions& options,
                        MeshPostProcessingStats* stats = nullptr);

// Reads a PLY mesh through the repository's CV_IO reader, applies the shared
// operation, and writes a binary PLY. If input and output are identical, the
// original is replaced only after the processed file has been written.
bool PostProcessMeshFile(const std::string& input_path,
                         const std::string& output_path,
                         const MeshPostProcessingOptions& options,
                         MeshPostProcessingStats* stats = nullptr);

}  // namespace mvs
}  // namespace colmap
