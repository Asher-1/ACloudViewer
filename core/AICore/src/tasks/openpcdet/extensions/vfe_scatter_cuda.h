#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace openpcdet::ext {

// Per-PFN-layer weights for the dynamic PillarVFE. All arrays are host,
// flat, and uploaded internally. linear_w is [out_C][in_C] row-major.
struct gpu_pfn_layer {
    std::vector<float> linear_w;   // [out_C * in_C]
    std::vector<float> linear_b;   // [out_C] (may be empty)
    std::vector<float> bn_scale;   // [out_C] (may be empty -> identity)
    std::vector<float> bn_bias;    // [out_C] (may be empty)
    int  in_channels  = 0;
    int  out_channels = 0;
    bool last_layer   = false;
};

// Runs the dynamic PillarVFE (voxelize + per-point MLP + per-pillar maxpool)
// and the PointPillarScatter on GPU. Mirrors run_dynamic_pillar_vfe_cpu +
// the scatter step in pointpillar.cpp exactly.
//
// Inputs:
//   points      - [n_points * point_dim] host point cloud
//   grid/voxel  - grid size, voxel size, point-cloud range, feature flags
//   layers      - PFN layers (host weights)
//   C_bev       - num_bev_features (== last layer out_channels)
//
// Outputs (host buffers, caller-owned):
//   pillar_coords    - [V * 4]  (0,0,gy,gx) per pillar, V pillars
//   pillar_features  - [V * C_bev] (maxpooled PFN output per pillar)
//                      when need_pillar_features is true; otherwise resized
//                      only, because the caller will not read the host values
//   spatial_whc      - [nx * ny * C_bev] BEV in ggml [W,H,C] layout (W fastest)
//
// Returns false and fills error on failure. If no points fall in the grid,
// returns true with V == 0 and spatial_whc zeroed.
bool gpu_pillar_vfe_scatter(const float * points, int32_t n_points, int32_t point_dim,
                            int nx, int ny, int nz,
                            float vx, float vy, float vz,
                            float x0, float y0, float z0,
                            bool use_abs, bool with_distance,
                            const std::vector<gpu_pfn_layer> & layers, int C_bev,
                            std::vector<int32_t> & pillar_coords,
                            std::vector<float> & pillar_features,
                            std::vector<float> & spatial_whc,
                            std::string & error,
                            float * spatial_dst_dev = nullptr,
                            bool need_pillar_features = true);

// Pre-voxelized variant (non-dynamic PillarVFE): takes voxels [V*P*point_dim]
// (zero-padded P dim), num_points[V], coords[V*4]=[b,z,y,x], V, P. Mirrors
// run_pillar_vfe_cpu + the scatter step. Outputs pillar_features [V*C_bev]
// and spatial_whc [nx*ny*C_bev] in ggml [W,H,C] layout.
bool gpu_pillar_vfe_scatter_vox(const float * voxels, const int32_t * num_points,
                                const int32_t * coords, int32_t V, int32_t P, int32_t point_dim,
                                int nx, int ny,
                                float vx, float vy, float vz,
                                float x0, float y0, float z0,
                                bool use_abs, bool with_distance,
                                const std::vector<gpu_pfn_layer> & layers, int C_bev,
                                std::vector<float> & pillar_features,
                                std::vector<float> & spatial_whc,
                                std::string & error,
                                float * spatial_dst_dev = nullptr,
                                bool need_pillar_features = true);

} // namespace openpcdet::ext
