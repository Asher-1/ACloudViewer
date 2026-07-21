// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "nested.hpp"

#include <algorithm>
#include <cmath>

namespace aicore {
namespace depth {

// torch.quantile linear interpolation on a (copied) vector.
float NestedAligner::quantile(std::vector<float> v, float q) {
    if (v.empty()) return 0.0f;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n == 1) return v[0];
    double pos = (double)q * (double)(n - 1);
    double lo_d = std::floor(pos);
    size_t lo = (size_t)lo_d;
    size_t hi = (size_t)std::ceil(pos);
    if (hi >= n) hi = n - 1;
    double frac = pos - lo_d;
    return (float)((double)v[lo] + frac * ((double)v[hi] - (double)v[lo]));
}

void NestedAligner::process_mono_sky(std::vector<float>& depth,
                                     const std::vector<float>& sky) {
    const size_t N = depth.size();
    if (N == 0 || sky.size() != N) return;
    size_t n_nonsky = 0, n_sky = 0;
    for (size_t i = 0; i < N; ++i) (sky[i] < 0.3f ? n_nonsky : n_sky)++;
    if (n_nonsky <= 10 || n_sky <= 10)
        return;  // matches the reference early-returns
    std::vector<float> depth_ns;
    depth_ns.reserve(n_nonsky);
    for (size_t i = 0; i < N; ++i)
        if (sky[i] < 0.3f) depth_ns.push_back(depth[i]);
    const float non_sky_max = quantile(depth_ns, 0.99f);  // NOT capped here
    for (size_t i = 0; i < N; ++i)
        if (!(sky[i] < 0.3f)) depth[i] = non_sky_max;
}

NestedOut NestedAligner::align(const AnyviewOut& any,
                               const MetricOut& metric,
                               int H,
                               int W) {
    const size_t N = (size_t)H * (size_t)W;
    NestedOut out;
    out.depth.assign(any.depth.begin(), any.depth.end());
    out.extrinsics = any.extrinsics;
    out.intrinsics = any.intrinsics;

    // --- 1) apply_metric_scaling: scale metric depth by anyview focal / 300
    // --- focal = (intrinsics[0,0] + intrinsics[1,1]) / 2 ; scale_factor const
    // = 300.
    const float focal = (any.intrinsics[0] + any.intrinsics[4]) * 0.5f;
    const float metric_scale = focal / 300.0f;
    std::vector<float> metric_depth(N);
    for (size_t i = 0; i < N; ++i)
        metric_depth[i] = metric.depth[i] * metric_scale;

    // --- 2) _apply_depth_alignment ---
    // non_sky_mask = sky < 0.3
    std::vector<unsigned char> non_sky(N);
    for (size_t i = 0; i < N; ++i) non_sky[i] = (metric.sky[i] < 0.3f) ? 1 : 0;

    // median_conf = quantile(depth_conf[non_sky], 0.5). 224^2 < 100000 -> no
    // sampling.
    std::vector<float> conf_ns;
    conf_ns.reserve(N);
    for (size_t i = 0; i < N; ++i)
        if (non_sky[i]) conf_ns.push_back(any.depth_conf[i]);
    const float median_conf = quantile(conf_ns, 0.5f);

    // align_mask = (conf >= median_conf) & non_sky & (metric_depth > 1e-2) &
    // (depth > 1e-3)
    const float min_depth_thresh = 1e-3f;
    const float min_metric_thresh = 1e-2f;
    double num = 0.0, den = 0.0;  // least_squares_scale_scalar(a=metric,
                                  // b=depth) = sum(a*b)/sum(b*b)
    for (size_t i = 0; i < N; ++i) {
        if (non_sky[i] && any.depth_conf[i] >= median_conf &&
            metric_depth[i] > min_metric_thresh &&
            any.depth[i] > min_depth_thresh) {
            const double a = metric_depth[i];
            const double b = any.depth[i];
            num += a * b;
            den += b * b;
        }
    }
    if (den < 1e-12) den = 1e-12;
    const float scale_factor = (float)(num / den);
    out.scale_factor = scale_factor;

    // output.depth *= scale_factor ; output.extrinsics[:,:,:3,3] *=
    // scale_factor
    for (size_t i = 0; i < N; ++i) out.depth[i] *= scale_factor;
    out.extrinsics[3] *= scale_factor;   // row0 col3 (translation x)
    out.extrinsics[7] *= scale_factor;   // row1 col3 (translation y)
    out.extrinsics[11] *= scale_factor;  // row2 col3 (translation z)

    // --- 3) _handle_sky_regions ---
    // non_sky recomputed from same metric sky (unchanged). non_sky_max =
    // min(quantile(depth[non_sky], 0.99), 200). Then sky pixels -> non_sky_max.
    std::vector<float> depth_ns;
    depth_ns.reserve(N);
    for (size_t i = 0; i < N; ++i)
        if (non_sky[i]) depth_ns.push_back(out.depth[i]);
    float non_sky_max = quantile(depth_ns, 0.99f);
    if (non_sky_max > 200.0f) non_sky_max = 200.0f;
    for (size_t i = 0; i < N; ++i)
        if (!non_sky[i]) out.depth[i] = non_sky_max;

    return out;
}

}  // namespace depth
}  // namespace aicore
