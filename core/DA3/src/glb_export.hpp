#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace da {

// Options controlling .glb (glTF-2.0 binary) export.
struct GlbOptions {
    // Maximum number of points retained. <=0 or >= the produced point count
    // keeps ALL points. NOTE: the reference downsamples with np.random.choice
    // (nondeterministic); for byte-faithful parity we keep ALL points (no
    // downsample) so the output is deterministic. Default is large enough to
    // keep everything for our typical scenes.
    int num_max_points = 1000000;
    // Whether to emit camera-frustum wireframes (a LINES primitive).
    bool show_cameras = true;
    // Camera wireframe scale as a fraction of the estimated scene diagonal.
    float camera_size = 0.03f;
    // GLB adaptive confidence threshold parameters (see get_conf_thresh):
    //   lower = percentile(conf, conf_thresh_percentile)
    //   upper = percentile(conf, ensure_thresh_percentile)
    //   thr   = min(max(conf_thresh, lower), upper)
    float conf_thresh = 1.05f;
    float conf_thresh_percentile = 40.0f;
    float ensure_thresh_percentile = 90.0f;
};

// Write a glTF-2.0 binary point cloud (POINTS primitive) plus optional camera
// frustums (LINES primitive) to `path`. Inputs mirror back_project():
//   depth, conf : N*H*W row-major (frame, row, col).
//   K           : per-frame 3x3 intrinsics, row-major.
//   ext         : per-frame 4x4 world-to-camera extrinsics, row-major.
//   images_u8   : per-frame pointer to H*W*3 RGB uint8 image.
// Geometry is aligned to the first camera in glTF coordinates and centered on
// the per-axis median of the point cloud (see
// _compute_alignment_transform_first_cam_glTF_center_by_points). Returns false
// on I/O error.
bool write_glb(const std::string& path,
               const std::vector<float>& depth,
               const std::vector<float>& conf,
               const std::vector<std::array<float, 9>>& K,
               const std::vector<std::array<float, 16>>& ext,
               const std::vector<const uint8_t*>& images_u8,
               int H, int W, int N, const GlbOptions& opt);

} // namespace da
