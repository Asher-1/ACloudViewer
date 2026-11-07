#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "tasks/rmbg/swin_backbone.hpp"


namespace rmbg {

// Explicit per-session graph options. Every field replaces one of the
// RMBG_* / GGML_VK_* environment variables the upstream port read at graph
// build time; defaults reproduce the historical "optimized" profile
// bit-for-bit (see aicore_rmbg_options in rmbg_capi.h for the C ABI).
struct GraphOptions {
    // Resolved math profile ("strict" | "optimized" | "fast" |
    // "unsafe-fast" | "default"), set by apply_profile_to_graph from
    // aicore_rmbg_options_set_math_profile. GraphBuilder bakes the profile's
    // Vulkan matmul-dispatch decision into the graph as output-name marks
    // (rmbg_scalar_* / rmbg_tc_* / neutral rmbg_mm_*); the ggml patch routes
    // by those names, replacing the former process-global env whitelist.
    std::string math_profile;
    // Vulkan data flow.
    bool vulkan_direct_conv = true;   // was RMBG_VK_DIRECT_CONV=1 (optimized)
    bool vulkan_qkv_layout = true;    // was RMBG_VK_QKV_LAYOUT (default on)
    bool vulkan_flash_attn = true;    // was RMBG_VK_FLASH_ATTN (F32 scalar)
    bool vulkan_flash_coop = false;   // was RMBG_VK_FLASH_ATTN=coop[N] opt-in
    int vulkan_flash_coop_stage = -1; // -1 = all stages; 0..3 = one stage
    bool vulkan_deform_project = false;  // was RMBG_VK_DEFORM_PROJECT (off)
    bool vulkan_deform_project_coop = false;  // "coop" spelling
    bool vk_f16_disabled = true;      // was GGML_VK_DISABLE_F16 (optimized)
    // CUDA data flow.
    bool strict_math = false;         // was RMBG_STRICT_MATH / TF32_OVERRIDE=0
    bool cuda_f16_gemm = false;       // was RMBG_CUDA_F16_GEMM (default off)
    int cuda_f16_min_stage = 2;       // was RMBG_CUDA_F16_MIN_STAGE
    bool cuda_nn_gemm = false;        // was RMBG_CUDA_NN_GEMM (default off)
    // Metal data flow: F16 weights keep every GEMM on the F16 matrix-unit
    // path (kernel_mul_mm_f16_f32/f16_f16); F32 weights run the scalar
    // dequantize_f32 load inside kernel_mul_mm_f32_f32, several times slower
    // for the short-K conv GEMMs and the Swin MLP/QKV projections.  On by
    // default because the F16 im2col + F16 conv weights already ship in the
    // Metal path and the contract tolerates the rounding.
    bool metal_f16_gemm = true;
};

class RmbgDeviceGraph {
public:
    RmbgDeviceGraph();
    ~RmbgDeviceGraph();
    RmbgDeviceGraph(const RmbgDeviceGraph &) = delete;
    RmbgDeviceGraph & operator=(const RmbgDeviceGraph &) = delete;

    bool init(ggml_backend_t backend, const WeightMap & weights, int input_size,
              const GraphOptions & options, std::string & err);
    bool forward(const std::vector<float> & input_nchw, std::vector<float> & alpha,
                 std::string & err);

    // ACV: real-time progress reporting.  `cb` is invoked from the Metal
    // encode threads as graph nodes are submitted (user-first signature,
    // mirroring aicore_rmbg_progress_fn); pass nullptr to clear.  Returns
    // true when the backend actually supports progress callbacks (Metal),
    // false otherwise — callers use this to fall back to busy-mode UI.
    bool set_progress_callback(void (*cb)(void *, int, int), void *user);

    // Encoder taps are exposed only for numerical validation of the graph path.
    bool forward_encoder(const std::vector<float> & input_nchw,
                         std::vector<float> & x1, std::vector<float> & x2,
                         std::vector<float> & x3, std::vector<float> & x4,
                         std::string & err);
    bool forward_swin_debug(const std::vector<float> & input_nchw,
                            std::vector<float> & patch_tokens,
                            std::vector<float> & block0_tokens,
                            std::vector<float> & stage0_tokens,
                            std::vector<float> & stage1_tokens,
                            std::vector<float> & stage2_tokens,
                            std::vector<float> & stage3_tokens,
                            std::string & err);
    bool forward_block0_debug(const std::vector<float> & input_nchw,
                              std::vector<float> & patch_pre_norm,
                              std::vector<float> & norm1,
                              std::vector<float> & window0,
                              std::vector<float> & attended_window0,
                              std::vector<float> & after_attention,
                              std::string & err);

    size_t compute_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rmbg
