// QuerySplat decoder compute graph for ggml (querysplat-cli).
//
// Consumes the VGGT-Omega encoder's cached-layer tokens (exported by the
// embedded mapggml runtime via RuntimeOptions::expose_tokens) plus the
// camera-derived Plucker rays, and produces the raw deconv/color Linear
// outputs; the final activation heads run host-side (activations_host).
//
// Layout contract (mirrors the map-anything-ggml conventions):
//   layer tokens: 4 blocks appended, block i is ggml ne
//                 {2*enc_embed_dim, Nagg, S} f32 == torch
//                 (S, Nagg, 2*enc_embed_dim) under the memory-order rules
//                 (ne[0] fastest); Nagg == patch_token_start + P, the first
//                 patch_token_start rows per view are camera+register.
//   rgb:          (S,3,H,W) f32, ImageNet-NORMALIZED (torch C-order == ne
//                 {W,H,3,S}); feeds patch_embed.
//   plucker:      (S,6,H,W) f32 == ne {W,H,6,S}; feeds patch_plucker_embed.
//   tokens:       {C, T, B} f32; attention q/k/v {dh,nh,T,B} permuted to
//                 {dh,T,nh,B} for ggml_flash_attn_ext (no mask, scale
//                 1/sqrt(dh)); flash output reshapes straight to
//                 {C,T,B} (head-major token vectors).
//   weights:      GGUF tensor keys are the SHORT names produced by
//                 convert_querysplat_to_gguf.py rename_key() (ggml caps
//                 tensor names at 64 chars; the verbatim state-dict keys run
//                 to 71). gguf-py reversed the torch shapes on write
//                 ({out,in} -> ne {in,out}).
//
// Constant folding: layer_weights = softmax(MLP(eye_token)) depends only on
// checkpoint parameters (eye_token is a frozen Parameter in every released
// inference path, including both TTO modes), so the four floats are computed
// host-side at build time and the eye->mlp->softmax branch never enters the
// graph.
#pragma once

#include "tasks/map_anything/map_anything_backend.hpp"


#include <map>
#include <memory>
#include <string>
#include <vector>

namespace qs {

struct DecoderConfig {
    int enc_embed_dim = 2048;
    int enc_num_heads = 16;
    int dec_depth = 12;
    int post_decoder_depth = 6;
    float mlp_ratio = 4.0f;
    int num_gs_tokens = 8192;
    int num_gaussians_per_token = 64;
    int patch_size = 16;  // vggt_original_encoder_patch_size
    float gaussian_scale_cap = 0.075f;
    float gaussian_z_offset = 1.0f;
    int gaussian_sh_degree = 0;
    float gaussian_sh_rest_scale = 0.1f;
    // official pointcloud.ply filter (options.gaussian_save_opacity_threshold);
    // independent of the CLI --opacity-threshold used for gaussians.ply
    float gaussian_save_opacity_threshold = 0.005f;
    float znear = 0.025f;
    float zfar = 125.0f;
    // options.img_size (model resolution H, W: infer.py / evaluate_jsons.py
    // preprocess, reconstruct and predict cameras at it) and
    // options.bg_color ("white" | "black" | "grey") as the rasterizer RGB
    int img_size[2] = {512, 512};
    float bg_color[3] = {0.5f, 0.5f, 0.5f};
    // host-folded layer weights (softmax over the 4 cached layers), filled
    // at init from eye_token/eye_layer_mlp
    float layer_weights[4] = {0.25f, 0.25f, 0.25f, 0.25f};
};

struct DecoderOutputs {
    // final gaussians, torch (N, 11+color_dim) C-order:
    // [pos(3) | opacity | scale(3) | rot(4) | color(color_dim)]
    std::vector<float> gaussians;
    int64_t N = 0;
    int color_dim = 3;
    // parity seams (filled only when QS_DUMP_STAGE is set at run time)
    std::map<std::string, std::vector<float>> seams;
};

struct DecoderRuntimeOptions {
    // Parity seam dump directory (mirrors map-anything-ggml's
    // RuntimeOptions::dump_dir). When non-empty, run() writes the decoder
    // seam tensors as raw f32 bins for scripts/cmp_querysplat.py. "" = off.
    std::string dump_dir;
    // F32-accuracy attention intent: on the CUDA backend the flash kernels
    // convert F32 K/V to F16 (~5e-4 relative per call x 18 blocks), so the
    // decoder builds attention from explicit F32 softmax/matmul ops instead.
    // No-op on CPU/Vulkan (their flash path is F32-exact). Off by default.
    bool f32_accuracy = false;
    // TTO: also fetch the four KV memories (geom K/V + post_rgb K/V) into
    // DecoderOutputs::seams ("cpp_geom_kv_k/v", "cpp_post_kv_k/v") without
    // writing any files. Used by tto_begin to seed the trainable memory.
    bool fetch_kv = false;
};

// KV-only read of the querysplat.* metadata (no tensor data): the model
// resolution is needed before any frame is decoded. False when the file cannot
// be opened or carries an unknown bg_color.
bool read_decoder_config(const std::string& gguf_path, DecoderConfig* cfg);

class QuerySplatDecoder {
   public:
    // Loads the GGUF (weights staged lazily on first build), parses the
    // querysplat.* metadata.
    QuerySplatDecoder(const std::string& gguf_path, map_anything::Backend backend,
                      const DecoderRuntimeOptions& opts = DecoderRuntimeOptions());
    ~QuerySplatDecoder();
    bool valid() const;

    const DecoderConfig& config() const { return cfg_; }

    // Builds/rebuilds the graph when (S, H, W, patch_token_start) changes.
    // patch_token_start comes from the encoder export (== 1 + num_register).
    bool ensure_shape(int S, int H, int W, int patch_token_start);

    // One inference. layer_tokens: 4 appended blocks as exported by
    // VGGTOutputs::cached_tokens; rgb: normalized (S,3,H,W); plucker:
    // (S,6,H,W). Returns the RAW deconv/color outputs activated host-side.
    bool run(const float* layer_tokens, const float* rgb, const float* plucker,
             DecoderOutputs& out);

    // ---- TTO session (t4; sh0 only) ------------------------------------
    // Fine-tunes the KV memory (geom K/V + post_rgb K/V) against the input
    // views' own pixels, mirroring QuerySplat.forward_tto_memory. The
    // session builds a dedicated graph pair (forward; backward+AdamW) with
    // the four memory tensors as trainable inputs; per-step flow:
    //   tto_forward  -> raw heads (host activation + render + loss backward
    //                   happen OUTSIDE, via t1/t2/t3)
    //   tto_backward -> consumes d(raw), runs the block backward + AdamW
    //   tto_gaussians-> current memory -> gaussians (host activation)
    bool tto_begin(int S, int H, int W, int patch_token_start,
                   const float* layer_tokens, const float* rgb,
                   const float* plucker, float lr);
    bool tto_forward(std::vector<float>& deconv_raw,
                     std::vector<float>& color_raw);
    bool tto_backward(const float* d_deconv, const float* d_color, int step,
                      std::vector<float>* dmem_out[4] = nullptr);
    // fd-parity helpers: raw access to the trainable memory tensors
    bool tto_mem_get(int idx, std::vector<float>& out);
    bool tto_mem_set(int idx, const std::vector<float>& in);
    bool tto_gaussians(DecoderOutputs& out);
    // host activation of ALREADY-fetched raw heads (no graph compute)
    DecoderOutputs tto_gaussians_from(const std::vector<float>& deconv_raw,
                                      const std::vector<float>& color_raw);
    void tto_end();

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    DecoderConfig cfg_;
};

}  // namespace qs
