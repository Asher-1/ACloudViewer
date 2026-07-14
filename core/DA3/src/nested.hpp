#pragma once
#include <array>
#include <vector>

namespace da {

// Anyview (GIANT) branch outputs for the nested metric alignment.
struct AnyviewOut {
    std::vector<float> depth;       // [H*W] row-major
    std::vector<float> depth_conf;  // [H*W] row-major
    std::array<float,12> extrinsics; // 3x4 row-major
    std::array<float,9>  intrinsics; // 3x3 row-major
};

// Metric (ViT-L + DPT+sky) branch outputs for the nested metric alignment.
struct MetricOut {
    std::vector<float> depth;  // [H*W] row-major (depth_metric_raw, pre metric-scaling)
    std::vector<float> sky;    // [H*W] row-major
};

// Final aligned metric-scale result.
struct NestedOut {
    std::vector<float> depth;        // [H*W] metric-scale, sky-filled
    std::array<float,12> extrinsics; // 3x4 row-major (translation scaled)
    std::array<float,9>  intrinsics; // 3x3 row-major (unchanged by alignment)
    float scale_factor = 0.0f;
};

// Host (pure std::vector) implementation of NestedDepthAnything3Net's metric
// alignment (da3.py + utils/alignment.py). Given the anyview giant branch
// {depth, depth_conf, extrinsics, intrinsics} and the metric branch {depth, sky},
// produces the final metric-scale depth + scaled extrinsics + scale_factor.
class NestedAligner {
public:
    static NestedOut align(const AnyviewOut& any, const MetricOut& metric, int H, int W);

    // torch.quantile (linear interpolation): sort ascending, pos = q*(n-1),
    // interpolate between floor/ceil. v is copied & sorted internally.
    static float quantile(std::vector<float> v, float q);

    // DepthAnything3Net._process_mono_sky_estimation: the metric branch's own
    // sky-fill applied inside da3_metric(x), BEFORE the nested alignment. Sets
    // sky pixels (sky>=0.3) to quantile(non_sky_depth, 0.99) (NOT capped at 200).
    // No-op unless there are >10 non-sky AND >10 sky pixels. Mutates depth.
    static void process_mono_sky(std::vector<float>& depth, const std::vector<float>& sky);
};

} // namespace da
