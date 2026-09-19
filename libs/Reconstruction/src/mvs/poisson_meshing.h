// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <filesystem>
#include <string>

namespace colmap {
namespace mvs {

struct PoissonMeshingOptions {
    // This floating point value specifies the importance that interpolation of
    // the point samples is given in the formulation of the screened Poisson
    // equation. The results of the original (unscreened) Poisson Reconstruction
    // can be obtained by setting this value to 0.
    double point_weight = 1.0;

    // This integer is the maximum depth of the tree that will be used for
    // surface reconstruction. Running at depth d corresponds to solving on a
    // voxel grid whose resolution is no larger than 2^d x 2^d x 2^d. Note that
    // since the reconstructor adapts the octree to the sampling density, the
    // specified reconstruction depth is only an upper bound.
    int depth = 13;

    // If specified, the reconstruction code assumes that the input is equipped
    // with colors and will extrapolate the color values to the vertices of the
    // reconstructed mesh. The floating point value specifies the relative
    // importance of finer color estimates over lower ones.
    double color = 32.0;

    // This floating point values specifies the value for mesh trimming. The
    // subset of the mesh with signal value less than the trim value is
    // discarded.
    double trim = 10.0;

    // The number of threads used for the Poisson reconstruction.
    int num_threads = -1;

    bool Check() const;
};

// Perform Poisson surface reconstruction and return true if successful.
bool PoissonMeshing(const PoissonMeshingOptions& options,
                    const std::filesystem::path& input_path,
                    const std::filesystem::path& output_path);

}  // namespace mvs
}  // namespace colmap
