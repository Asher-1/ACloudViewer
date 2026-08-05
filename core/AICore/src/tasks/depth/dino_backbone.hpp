#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include "rope2d.hpp"
#include <vector>
namespace aicore {
namespace depth {
class DinoBackbone {
public:
    DinoBackbone(ModelLoader& ml, Backend& be) : ml_(ml), be_(be) {}
    // input_chw: [3,H,W] normalized. Returns tokens flattened [embed * (1+N_patch)],
    // ggml-order (embed fastest): element (token,e) at token*embed + e.
    bool prepare_tokens(const std::vector<float>& input_chw, int H, int W, std::vector<float>& out_tokens);
    bool forward(const std::vector<float>& input_chw, int H, int W,
                 std::vector<std::vector<float>>& feats,
                 std::vector<std::vector<float>>& cam_tokens);  // implemented in T15 - leave declared
    // Multi-view (S>=1) backbone with cross-view global attention.
    //   views_chw : S items, each [3,H,W] normalized.
    //   feats     : [L][S] each [256*1536] (cat[local_x, norm(x)], token0 stripped, per view).
    //   cam_tokens: [L][S] each [1536]     (cat[local_x[t0], x[t0]] RAW, per view).
    // For S >= THRESH_FOR_REF_SELECTION (=3) and no input cam-token, a reference
    // view is selected (saddle_balanced) at layer alt_start-1, views are reordered
    // (ref first), processed, and the ORIGINAL view order is restored before feats/
    // cam_tokens are produced. out_b_idx (optional) receives the selected reference
    // view index (0 when no selection is applied, e.g. S<3).
    bool forward_mv(const std::vector<std::vector<float>>& views_chw, int H, int W,
                    std::vector<std::vector<std::vector<float>>>& feats,
                    std::vector<std::vector<std::vector<float>>>& cam_tokens,
                    int* out_b_idx = nullptr);
    // Fused single-image path: build the SAME block loop as forward() but, at each
    // out-layer, produce the feat tensor IN-GRAPH instead of capturing to host:
    //   feat = ggml_concat([ local_x, layernorm(x, vit.norm.{w,b}, ln_eps) ], dim0)
    //          with token-0 stripped -> [2*embed, Npatch] (ne0=channel, ne1=token).
    // This matches forward()'s host post-process (cat[local_x_raw, norm(x)]) so the
    // features never leave the device. out_feat[o] for o in [0, out_layers.size()).
    // Restricted to cat_token=true (BASE/giant); returns false otherwise. Built into
    // the caller's compute() ctx/pool (engine fuses this with the DPT head graph).
    bool build_feats_graph(ggml_context* ctx, const std::vector<float>& input_chw,
                           int H, int W, GraphInputPool& pool, ggml_tensor* out_feat[4]);
private:
    std::vector<float> interp_pos_embed(int gh, int gw) const;  // host bicubic -> [(1+gh*gw)*embed], token-major embed-minor
    // Full multi-view forward on views already in processing order (ref first when
    // selection is applied). Returns feats/cam_tokens in the SAME (input) order.
    bool forward_mv_ordered(const std::vector<std::vector<float>>& views_chw, int H, int W,
                            std::vector<std::vector<std::vector<float>>>& feats,
                            std::vector<std::vector<std::vector<float>>>& cam_tokens);
    // Pass-A selection helper: run the per-view LOCAL blocks [0, upto) and capture
    // each view's token-0 (cls) feature (un-normalized) -> cls_out[s] = [embed].
    bool capture_local_cls(const std::vector<std::vector<float>>& views_chw, int H, int W,
                           int upto, std::vector<std::vector<float>>& cls_out);
    // saddle_balanced reference-view selection from per-view cls features.
    int select_reference_view_saddle(const std::vector<std::vector<float>>& cls, int embed) const;
    ModelLoader& ml_; Backend& be_;
};
} // namespace depth
} // namespace aicore
