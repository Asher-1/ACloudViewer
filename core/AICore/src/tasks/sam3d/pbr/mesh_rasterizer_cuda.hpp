// Optional CUDA implementation backed by the official nvdiffrast raster core.
#pragma once

#include <string>
#include <vector>

#include "mesh_rasterizer.hpp"

namespace sam3d {

bool mesh_visibility_frequency_nvdiffrast_cuda(
    const NativeMesh& mesh, const std::vector<MeshRasterCamera>& cameras,
    const MeshRasterConfig& config, std::vector<float>& frequency, std::string& error);

// Builds the official _fill_holes() Hammersley view matrices on CUDA. Keeping
// the trigonometry, cross products, normalization, and clip-space GEMMs on the
// device avoids a host-libm rounding boundary in the default native pipeline.
bool mesh_visibility_hammersley_frequency_nvdiffrast_cuda(
    const NativeMesh& mesh, int view_count, const MeshRasterConfig& config,
    std::vector<float>& frequency, std::string& error);

// Diagnostic contract used by the PyTorch parity harness. Values are row-major
// [view_count, 4, 4] matrices after device construction.
bool mesh_hammersley_camera_matrices_nvdiffrast_cuda(
    int view_count, std::vector<float>& views, std::vector<float>& projections, std::string& error);

}  // namespace sam3d
