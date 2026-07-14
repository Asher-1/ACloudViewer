#pragma once
#include <vector>
#include <array>

namespace da {

// World-space 3D Gaussians produced by the GaussianAdapter (host geometry).
// All arrays are flattened row-major; N = H*W (single view, B=V=1).
struct Gaussians {
    int N = 0;
    std::vector<float> means;      // [N*3]        (pixel*3 + xyz)
    std::vector<float> scales;     // [N*3]        (pixel*3 + xyz)
    std::vector<float> rotations;  // [N*4]   wxyz (pixel*4 + c)
    std::vector<float> harmonics;  // [N*3*9]      (pixel*3 + color)*9 + coeff
    std::vector<float> opacities;  // [N]
};

// GaussianAdapter: pure host math (NO learned weights). Converts the GSDPT
// raw gaussian channels (37) + confidence + the giant depth + camera pose into
// world-space 3D Gaussians, exactly mirroring depth_anything_3.model.gs_adapter
// .GaussianAdapter.forward (sh_degree=2, pred_offset_depth/xy, no pred_color).
class GsAdapter {
public:
    int   sh_degree         = 2;
    int   d_sh              = 9;     // (sh_degree+1)^2
    bool  pred_offset_depth = true;
    bool  pred_offset_xy    = true;
    bool  pred_color        = false;
    float scale_min         = 1e-5f;
    float scale_max         = 30.0f;

    // raw_gs  : [H*W*37], channels-LAST: (h*W+w)*37 + c.
    // depth   : [H*W], row-major (h*W+w).
    // gs_conf : [H*W], row-major; mapped to opacity via map_pdf_to_opacity.
    // ext     : 3x4 row-major, the world->camera extrinsics (w2c).
    // intr    : 3x3 row-major camera intrinsics K (pixel units, H/W image).
    // Returns false on malformed input (size / singular intrinsics).
    bool build(const std::vector<float>& raw_gs, const std::vector<float>& depth,
               const std::vector<float>& gs_conf,
               const std::array<float,12>& ext, const std::array<float,9>& intr,
               int H, int W, Gaussians& out) const;
};

} // namespace da
