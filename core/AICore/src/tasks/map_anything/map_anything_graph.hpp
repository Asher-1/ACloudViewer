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

#include "tasks/map_anything/map_anything_backend.hpp"

#include "tasks/map_anything/map_anything_gguf_loader.hpp"


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
    // Token-consumer support (querysplat-style decoder on top of the
    // aggregator): pin all cached-layer outputs and export them through
    // VGGTOutputs::cached_tokens. Costs one set_output pin per cached layer
    // (~2C*Nagg*S*4B live working set each). Off by default; honored by the
    // vggt_omega builder (querysplat consumes the vggt-omega 1B/512 GGUFs).
    bool expose_tokens = false;
    // GPU f32-accuracy parity profile (torch allow_tf32=false semantics):
    // build attention from explicit F32 ops instead of ggml_flash_attn_ext
    // (whose CUDA kernels convert F32 K/V to F16) and dequantize
    // Q8_0/Q4_x/Q5_x Linear weights to F32 per call on CPU/CUDA (those
    // backends' quantized GEMMs quantize the ACTIVATION too). CPU/Vulkan
    // flash attention is already F32-exact; on those backends only the
    // legacy-quant weight path matters. Off by default — map_anything's own
    // behavior is unchanged; the querysplat task enables it per options.
    bool f32_accuracy = false;
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
    // Token-consumer support (RuntimeOptions::expose_tokens): the
    // aggregator's cached-layer outputs {2*embed_dim, Nagg, S} f32, appended
    // per layer in cached_layer_idx order (each block contiguous, ne[0]
    // fastest). Per layer this is torch (S, Nagg, 2*embed_dim) under the
    // repo's memory-order rules. Empty unless requested at construction.
    std::vector<float> cached_tokens;
    // The cached layer indices behind cached_tokens (e.g. {4, 11, 17, 23}).
    std::vector<int> cached_layer_idx;
    // Tokens per view ahead of the patch rows: [0] = camera token,
    // [1 .. patch_token_start) = register tokens (patch_token_start
    // == 1 + num_register_tokens). The final cached layer's first
    // patch_token_start rows are the camera/register tokens; patch rows
    // start there in every cached layer.
    int patch_token_start = 0;
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
    // When !valid(): true if the device could not hold this geometry (the
    // caller may retry with fewer views), and the reason ("" if unknown).
    bool out_of_memory() const { return out_of_memory_; }
    const std::string& error() const { return error_; }

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
    bool out_of_memory_ = false;
    std::string error_;
};

}  // namespace map_anything
