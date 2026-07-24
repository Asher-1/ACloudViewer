#pragma once
#include "aicore/export.h"
#include "ggml.h"
#include "gguf.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace aicore {
namespace depth {
class Backend;

struct Config {
    uint32_t patch_size = 14;
    uint32_t embed_dim = 0, depth = 0, num_heads = 0, head_dim = 0, mlp_hidden = 0;
    uint32_t num_register = 0, pos_embed_grid = 0;
    int32_t  alt_start = -1, rope_start = -1, qknorm_start = -1;
    float    init_values = 0.f, rope_freq = 100.f, ln_eps = 1e-6f, interp_offset = 0.1f;
    bool     cat_token = true, qkv_bias = true, interp_antialias = false;
    bool     head_pos_embed = true;         // DPT head UV pos-embed (metric DPT: false)
    std::string ffn_type = "mlp";           // "mlp" | "swiglu"
    uint32_t head_features = 0;             // DPT head feature width
    std::vector<int32_t> out_layers;
    std::vector<int32_t> head_out_channels; // DPT head per-stage out_channels
    std::vector<float>   img_mean, img_std;
    uint32_t    img_resize_target = 504;          // longest/shortest-side processing resolution
    std::string img_resize_mode = "upper_bound";  // "upper_bound" | "lower_bound"
    std::string checkpoint_name;
    float       head_max_depth = 0.f;             // DA2 metric scale (0 = relative)
    std::string arch = "depthanything3";          // route discriminator ("depthanything2" = DA2)
};

class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader();
    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;

    bool load(const std::string& path);
    const Config& config() const { return cfg_; }
    void set_img_resize_target(uint32_t target) { cfg_.img_resize_target = target; }
    ggml_tensor* tensor(const std::string& name) const;
    bool weights_on_device() const { return device_ctx_ != nullptr; }
    bool offload_weights(Backend& be);
    void release_gpu_weights();
private:
    void rebuild_host_tensor_map();
    Config cfg_;
    gguf_context* gguf_ = nullptr;
    ggml_context* ctx_  = nullptr;
    std::unordered_map<std::string, ggml_tensor*> tensors_;
    ggml_context* device_ctx_ = nullptr;
    ggml_backend_buffer* gpu_buf_ = nullptr;
};
} // namespace depth
} // namespace aicore
