// Host-side activation heads: turn the decoder graph's raw Linear outputs
// into the final (N, 11+color_dim) gaussian buffer. Ports
// scripts/models/activations.py PartialClipActivationHead and QuerySplat's
// _post_output/_activate_post_output exactly, in f32.
//
// Why host-side: sign/expm1/clamp/L2-normalize are a handful of elementwise
// passes over ~30 MB — milliseconds — and plain C++ f32 is bit-testable
// against torch, sidestepping ggml op availability (expm1/sign/clamp) and
// the expm1-under-f32 precision trap. They can move in-graph later if a
// workload ever demands it.
//
// Layout contract:
//   geom_raw:  deconv Linear output, torch (n_tokens, 640) C-order == ggml
//              {640, n_tokens}; row j = p*10 + c (p = gaussian-in-token,
//              c = attr). NOTE: the column count is the TOKEN count
//              (num_gs_tokens), not the gaussian count.
//   color_raw: delta head output, torch (n_tokens, post_output_dim*64)
//              C-order == ggml {post_output_dim*64, n_tokens}; row j =
//              p*post_output_dim + c.
//   gaussians: torch (N, 11+color_dim) == [pos(3) | opacity | scale(3) |
//              rot(4) | color(color_dim)], N = n_tokens * gpt.
#pragma once

#include <cstdint>
#include <vector>

namespace qs {

struct GaussianOutput {
    int64_t N = 0;
    int color_dim = 3;              // 3 (sh degree 0) or 12 (degree 1)
    std::vector<float> data;        // N * (11 + color_dim)
    // per-gaussian opacity, kept alongside for the PLY opacity filter
    std::vector<float> opacity;     // N
};

GaussianOutput activate_gaussians(const float* geom_raw,
                                  const float* color_raw, int64_t n_tokens,
                                  int gpt, int post_output_dim,
                                  float scale_cap, float z_offset,
                                  int sh_degree, float sh_rest_scale);

// Exact VJP of activate_gaussians (TTO t2): d_gaussians (N, 11+color_dim,
// the render_backward output layout) -> d_geom_raw (n_tokens*10*gpt) and
// d_color_raw (n_tokens*post_output_dim*gpt), the raw Linear-output grads.
// Torch-semantics corner cases are kept: sign/abs at x==0 give 0 gradient,
// the scale min() saturates to 0 at the cap (inclusive), F.normalize
// projects the gradient when the norm is realizable.
void activate_gaussians_backward(const float* geom_raw, const float* color_raw,
                                 const float* d_gaussians, int64_t n_tokens,
                                 int gpt, int post_output_dim, float scale_cap,
                                 float z_offset, int sh_degree,
                                 float sh_rest_scale, float* d_geom_raw,
                                 float* d_color_raw);

}  // namespace qs
