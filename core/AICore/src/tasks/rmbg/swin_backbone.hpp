#pragma once
#include "ggml.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rmbg {

struct WeightTensor {
    enum ggml_type type = GGML_TYPE_F32;
    std::vector<uint8_t> bytes;
    std::vector<int64_t> shape;
};

struct WeightMap {
    std::unordered_map<std::string, WeightTensor> tensors;
    mutable std::unordered_map<std::string, std::vector<float>> f32_cache;

    bool load_gguf(const char * path, std::string & err);
    bool merge_gguf(const char * path, std::string & err);
    const WeightTensor * get_tensor(const char * key) const;
    const std::vector<float> * get_f32(const char * key) const;
    const std::vector<int64_t> * get_shape(const char * key) const;
};

struct SwinConfig {
    int embed_dim   = 192;
    int num_heads   = 6;
    int window_size = 12;
    int mlp_ratio   = 4;
    float eps       = 1e-5f;
};

/// Patch merge between Swin stages (4C layer-norm + linear -> 2C).
struct PatchMergeWeights {
    int in_dim = 0;
    std::vector<float> nw, nb, red_w;
};

/// Per-block weights (W-MSA / SW-MSA).
struct SwinBlockWeights {
    int dim = 192;
    int num_heads = 6;
    int shift_size = 0;
    std::vector<float> n1w, n1b, n2w, n2b;
    std::vector<float> qkv_w, qkv_b, proj_w, proj_b;
    std::vector<float> rpb_table;
    std::vector<int>   rel_pos_idx;
    std::vector<float> mlp1_w, mlp1_b, mlp2_w, mlp2_b;
};

/// patch_embed + Swin stage-0 (2 blocks: shift 0 + shift ws/2).
class SwinBackboneForward {
public:
    bool init(const WeightMap & w, std::string & err);

    bool forward_patch_embed(const std::vector<float> & nchw_in, int H, int W,
                             std::vector<float> & patch_embed_out,
                             std::vector<float> & tokens_blc,
                             std::string & err) const;

    bool forward_stage0(const std::vector<float> & nchw_in, int H, int W,
                        std::vector<float> & patch_embed_out,
                        std::vector<float> & stage0_out,
                        std::string & err) const;

    /// Patch merge between stages. merge_idx 0: stage0->1, 1: stage1->2, ...
    bool forward_patch_merge(const std::vector<float> & tokens_blc, int H, int W, int merge_idx,
                             std::vector<float> & out, int & out_h, int & out_w,
                             std::string & err) const;

    /// stage0 + patch_merge + stage1 (2 blocks). Requires >=4 blocks loaded.
    bool forward_stage01(const std::vector<float> & nchw_in, int H, int W,
                         std::vector<float> & patch_embed_out,
                         std::vector<float> & stage0_out,
                         std::vector<float> & merge_out,
                         std::vector<float> & stage1_out,
                         std::string & err) const;

    /// Through Swin stage-2 (18 blocks @ 64²). Requires merges_[0,1] + 22 blocks.
    bool forward_stage2(const std::vector<float> & nchw_in, int H, int W,
                        std::vector<float> & patch_embed_out,
                        std::vector<float> & stage2_out,
                        std::string & err) const;

    /// Through Swin stage-3 (2 blocks @ 32²). Requires merges_[0,1,2] + 24 blocks.
    bool forward_stage3(const std::vector<float> & nchw_in, int H, int W,
                        std::vector<float> & patch_embed_out,
                        std::vector<float> & stage3_out,
                        std::string & err) const;

    /// Swin bb(x): four stage outputs after bb_norm0–3, NCHW [192,384,768,1536] @ 256²…32².
    bool forward_bb_four_scales(const std::vector<float> & nchw_in, int H, int W,
                                std::vector<float> & x1, std::vector<float> & x2,
                                std::vector<float> & x3, std::vector<float> & x4,
                                std::string & err) const;

    bool has_bb_norm() const { return has_bb_norm_; }

    bool forward_block(const std::vector<float> & tokens_blc, int H, int W,
                       int block_idx, std::vector<float> & out, std::string & err) const;

    bool window_attention_only(int block_idx, std::vector<float> x_nxc,
                               std::vector<float> & out, std::string & err) const;

    const SwinConfig & config() const { return cfg_; }
    bool has_stage1() const { return blocks_.size() >= 4 && merges_.size() >= 1; }
    bool has_stage2() const { return blocks_.size() >= 22 && merges_.size() >= 2; }
    bool has_stage3() const { return blocks_.size() >= 24 && merges_.size() >= 3; }
    int num_blocks() const { return (int) blocks_.size(); }

private:
    SwinConfig cfg_;
    std::vector<float> pe_w_, pe_b_, pe_nw_, pe_nb_;
    std::vector<PatchMergeWeights> merges_;
    std::vector<SwinBlockWeights> blocks_;
    int pe_out_c_ = 0;
    bool has_bb_norm_ = false;
    std::vector<float> bb_norm_w_[4], bb_norm_b_[4];

    static void tokens_to_nchw(const std::vector<float> & tokens, int H, int W, int C,
                               std::vector<float> & nchw);
    static void conv2d_nchw(const std::vector<float> & in, int N, int C, int H, int W,
                            const std::vector<float> & w, int OC, int KH, int KW,
                            int stride, const std::vector<float> * bias,
                            std::vector<float> & out);
    static void layer_norm(std::vector<float> & x, int rows, int cols,
                           const std::vector<float> & gamma, const std::vector<float> & beta, float eps);
    static void linear(const std::vector<float> & in, int M, int K, int N,
                       const std::vector<float> & w, const std::vector<float> * b,
                       std::vector<float> & out);
    static float gelu(float x);
    static void spatial_roll(std::vector<float> & feat, int B, int H, int W, int C, int dh, int dw);
    static void build_shift_attn_mask(int H, int W, int ws, int shift_size,
                                      std::vector<float> & mask_nW_N_N);

    static bool load_block_weights(const WeightMap & w, const std::string & prefix,
                                    int shift_size, SwinBlockWeights & out, std::string & err);

    static bool load_patch_merge_weights(const WeightMap & w, int stage,
                                         PatchMergeWeights & pm, std::string & err);

    static bool try_load_stage_blocks(const WeightMap & w, int stage, int depth, int ws,
                                        std::vector<SwinBlockWeights> & out, std::string & err);

    void window_attention(std::vector<float> & x, int BnW, int N, int C,
                          const SwinBlockWeights & bw,
                          const std::vector<float> * shift_mask,
                          int mask_win_idx = 0) const;

    void swin_block(std::vector<float> & tokens, int B, int H, int W,
                    const SwinBlockWeights & bw) const;
};

// Legacy alias (block0-only API used by early tests).
using SwinBlock0Forward = SwinBackboneForward;

} // namespace rmbg
