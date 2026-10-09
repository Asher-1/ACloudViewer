// VGGT-Omega compute graph for ggml (map_anything).
//
// Layout contract (official vggt-omega architecture, patch 16):
//   images:   {W, H, 3, S}  f32 in [0, 1]; the aggregator-internal ResNet
//             mean/std is applied as the first graph op
//   tokens:   {C, T, B}  f32, C contiguous per token
//   attention q/k/v: {dh, nh, T, B} views of the qkv projection, permuted to
//             {dh, T, nh, B} for ggml flash_attn_ext (no mask)
//   rope:     DINOv3-style axial rope with normalized coords; per-token
//             cos/sin rows [64] are precomputed on the host (identity rows
//             for camera/register tokens) and fed as {64, 1, T, 1} inputs
//   feature maps: {W', H', C', B} as required by ggml_conv_2d
//
// The graph is rebuilt whenever (S, H, W) changes; weights are staged once.
#pragma once

#include "map_anything_backend.hpp"
#include "map_anything_gguf_loader.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace map_anything {

// Explicit runtime switches. Everything that used to travel through
// environment variables (MAP_ANYTHING_DUMP_STAGE / MAPGGML_DOT) goes through
// this struct, so behavior is reproducible from the call site alone.
struct RuntimeOptions {
    // Stage-dump directory (only effective in builds compiled with
    // MAP_ANYTHING_ENABLE_DUMP=ON): after each run() the selected intermediate
    // tensors are written as <dir>/*.bin for parity debugging. "" = off.
    std::string dump_dir;
    // Optional path for ggml_graph_dump_dot (graph-shape debugging aid).
    // "" = off.
    std::string dot_path;
};

struct VGGTOutputs {
    // pose_enc {S, 9} f32: absT 3 + quaR 4 + FoV 2 (fov passed through
    // relu(x) + 0.01, translation/quaternion linear)
    std::vector<float> pose_enc;
    // dense head outputs {S, H, W}: depth = exp(logits),
    // depth_conf = 1 + exp(logits)
    std::vector<float> depth;
    std::vector<float> depth_conf;
    // Original-VGGT point head outputs {S, H, W, 3} / {S, H, W}
    // (empty for vggt-omega, which has no point head)
    std::vector<float> world_points;
    std::vector<float> world_points_conf;
    // Pi3 outputs (empty for the VGGT builders). camera_poses are the
    // camera-to-world 4x4 matrices (row-major, S*16) after the host-side
    // SVD orthogonalization of the raw head output; pose_raw is the
    // pre-SVD fc_t|fc_rot stack (S*12); conf holds RAW logits (the
    // official model documents sigmoid() for probabilities).
    std::vector<float> camera_poses;   // {S, 4, 4}
    std::vector<float> pose_raw;       // {S, 12} (pi3) / {S, 7} (mapanything)
    std::vector<float> local_points;   // {S, H, W, 3}
    std::vector<float> conf;           // {S, H, W}
    // MapAnything outputs (empty for the other builders). mask is the
    // sigmoid probability; metric_scale = exp(scale-token head).
    std::vector<float> mask;           // {S, H, W}
    float metric_scale = 0.0f;
    // vggt-omega text alignment (enable_alignment=True checkpoints): the
    // L2-normalized language-aligned embedding {2048} read out of the
    // final cached layer (empty when the GGUF carries no text head)
    std::vector<float> text_embedding;
    // Geometry of the dense outputs. vggt-omega produces the input
    // resolution (pixel-shuffle head); original-VGGT produces the
    // patch-aligned resolution (Hp*patch, Wp*patch), which differs from the
    // input whenever H/W are not multiples of the patch size.
    int out_h = 0;
    int out_w = 0;
    // wall-clock timings (ms) filled by run(): "build_graphs",
    // "stage1_backbone", "stage2_aggregator", "stage3_heads",
    // "inference_total" (used by the CLI --timing JSON)
    std::map<std::string, double> timing_ms;
};

class VGGTRuntime {
   public:
    // Does not take ownership of the GGUFModel; it must outlive the runtime.
    VGGTRuntime(const GGUFModel& model, Backend backend, int n_views,
                int height, int width,
                const RuntimeOptions& opts = RuntimeOptions());
    ~VGGTRuntime();

    // images_f32: S*3*H*W floats in [0,1], torch (S,3,H,W) order.
    // Returns false and logs on failure.
    bool run(const float* images_f32, VGGTOutputs& out);

    // False when construction failed (unknown architecture / init error);
    // run() would return false as well.
    bool valid() const { return impl_ != nullptr; }

    int n_views() const { return s_; }
    int height() const { return h_; }
    int width() const { return w_; }

    // Public nested types: Impl is the vggt-omega builder (registered as
    // "vggt_omega"); VGGTImpl is the original-VGGT builder ("vggt");
    // Pi3Impl is the Pi3 builder ("pi3"). All implement IGraphBuilder
    // (see graph_builder.hpp).
    struct Impl;
    struct VGGTImpl;
    struct Pi3Impl;
    struct Pi3XImpl;
    struct MapAnythingImpl;
    struct Dust3rImpl;

   private:
    std::unique_ptr<Impl> impl_;
    int s_ = 0, h_ = 0, w_ = 0;
};

}  // namespace map_anything
