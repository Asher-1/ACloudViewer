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
