// 3D Gaussian Splatting rasterizer (CPU reference), aligned with
// gsplat.rasterization semantics as invoked by scripts/rendering/gs.py:
//   - viewmats = world-to-camera (QuerySplat passes cam_view^T == w2c)
//   - K from intrinsics [fx, fy, cx, cy]
//   - EWA projection with the classic 0.3 low-pass dilate (antialiasing off)
//   - near/far plane culling, opacity-aware bbox extent
//   - tile (16x16) gaussian bins sorted by depth, front-to-back alpha
//     blending; a gaussian whose next_T would drop to <= 1e-4 is EXCLUDED
//     (not blended) and the pixel is done — gsplat
//     RasterizeToPixels3DGSFwd.cu semantics, with the per-pixel last-k index
//     recorded for the backward pass
//   - render mode "RGB+ED": RGB blended with the background at the tail
//     transmittance; depth = alpha-weighted expected depth with the
//     gsplat render_alphas.clamp(min=1e-10) denominator
//   - sh degree 0: colors used directly; degree 1: per-view SH evaluation
//
// render_backward() is the exact VJP of render_gaussians() for sh_degree 0
// (TTO): d_images -> d[pos3 | op | scale3 | rot4 | color3], porting
// RasterizeToPixels3DGSBwd.cu + persp_proj_vjp + inverse_vjp +
// quat_scale_to_covar_vjp from gsplat 1.5.3 (glm col-major translated to
// row-major math).
#pragma once

#include "tasks/querysplat/querysplat_activations.hpp"


#include <cstdint>
#include <string>
#include <vector>

namespace qs {

struct RenderConfig {
    int width = 512;
    int height = 512;
    float near_plane = 0.025f;
    float far_plane = 125.0f;
    float bg_color[3] = {0.5f, 0.5f, 0.5f};  // QuerySplat bg_color "grey"
    int sh_degree = 0;
};

struct RenderedViews {
    int S = 0;
    int width = 0, height = 0;
    std::vector<float> images;  // (S, 3, H, W) torch C-order [0,1]-ish
    std::vector<float> alphas;  // (S, 1, H, W)
    std::vector<float> depths;  // (S, 1, H, W)
    // per-pixel flat index of the last blended splat of the pixel's tile
    // (gsplat last_ids; -1 when nothing blended). Needed by render_backward.
    std::vector<int32_t> last_k;
};

// Gradients w.r.t. the activated gaussian buffer [pos(3) | op | scale(3) |
// rot(4) | color(color_dim)], the exact input layout of GaussianOutput.
struct GaussianGrads {
    int64_t N = 0;
    int color_dim = 3;
    std::vector<float> pos;      // N * 3
    std::vector<float> opacity;  // N
    std::vector<float> scale;    // N * 3
    std::vector<float> rot;      // N * 4
    std::vector<float> color;    // N * color_dim
};

// w2c: S*16 row-major world-to-camera; intrinsics: S*4 [fx, fy, cx, cy].
RenderedViews render_gaussians(const GaussianOutput& g, const float* w2c,
                               const float* intrinsics, int S,
                               const RenderConfig& cfg);

// Exact VJP of render_gaussians (sh_degree 0 only; d_images is (S,3,H,W)
// torch C-order, dL/dalphas and dL/ddepths are zero — the TTO loss does not
// touch them). fwd must be the same-step render_gaussians output.
bool render_backward(const GaussianOutput& g, const float* w2c,
                     const float* intrinsics, int S, const RenderConfig& cfg,
                     const RenderedViews& fwd, const float* d_images,
                     GaussianGrads* out);

// Plane-layout (3,H,W) float view in [0,1] -> interleaved RGB8 (H*W*3),
// the exact byte conversion infer.py applies ((clamp(0,1) * 255).astype(
// uint8), truncation not rounding). PNG encoding stays in the Qt/plugin
// layer — the task only hands over typed buffers.
void render_to_rgb8(const float* rgb, int width, int height,
                    unsigned char* out_rgb8);

}  // namespace qs
