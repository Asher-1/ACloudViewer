#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include <vector>

namespace da {

// Full DualDPT MAIN depth path (DA3) as a ggml forward, taking the four backbone
// out-layer features (feat_5/7/9/11) as input and producing the dense depth +
// confidence maps. Mirrors DualDPT._forward_impl (main branch only).
class DptHead {
public:
    DptHead(ModelLoader& ml, Backend& be) : ml_(ml), be_(be) {}

    // feats: 4 vectors each [N*C] = [256*1536], token-major / channel-minor
    //        (flat index = token*C + channel), i.e. the backbone out-layer features.
    // Returns depth [H*W] and conf [H*W] (row-major h,w; w fastest).
    bool depth(const std::vector<std::vector<float>>& feats, int H, int W,
               std::vector<float>& depth_out, std::vector<float>& conf_out);

    // Same as depth() but also reads back the 4 post-resize stage maps and the
    // post-output_conv1 fused map for the layer-isolation parity gates.
    //   stages[s]: [out_channels[s] * Hs * Ws] (ggml [W,H,C] order)
    //   fused    : [64 * 128 * 128]            (ggml [W,H,C] order)
    bool depth_debug(const std::vector<std::vector<float>>& feats, int H, int W,
                     std::vector<float>& depth_out, std::vector<float>& conf_out,
                     std::vector<std::vector<float>>& stages, std::vector<float>& fused);

    // Metric single-head: dim_in = embed (cat_token false), NO head.norm (norm_type
    // "idt"), output_dim 1 (depth only) + a parallel sky head. depth = exp(logit),
    // sky = relu(logit). feats: 4 vectors each [N*embed]. Returns depth & sky [H*W].
    bool depth_sky(const std::vector<std::vector<float>>& feats, int H, int W,
                   std::vector<float>& depth_out, std::vector<float>& sky_out);

    // DA2 relative/metric depth: single-channel DPT (output_dim==1), no UV pos-embed,
    // no head.norm. The two DA2 heads differ in the final activation:
    //   relative (max_depth<=0): depth = relu(logit)            (output_conv2 ReLU + F.relu)
    //   metric   (max_depth>0):  depth = sigmoid(logit)*max_depth (output_conv2 Sigmoid * max_depth)
    // No confidence/sky. feats: 4 vectors each [N*embed].
    bool depth_relative(const std::vector<std::vector<float>>& feats, int H, int W,
                        float max_depth, std::vector<float>& depth_out);

    // Build the DPT depth graph from the 4 backbone out-layer feat tensors (each
    // [C, N=Npatch], ne0=channel fastest) and return the logits tensor
    // [W,H,output_dim,1]. Shared by run() (unfused: feats uploaded as graph inputs)
    // and the fused single-image path (feats produced in-graph by the backbone, so
    // they never leave the device). Optional capture dests (array[4] stage_caps,
    // fused_cap, sky_cap) mirror run()'s debug captures; pass nullptr to skip. The
    // caller applies the host exp()/exp()+1 post-processing on the returned logits.
    ggml_tensor* build_depth_graph(ggml_context* ctx, ggml_tensor* const feat[4],
                                   int H, int W, GraphInputPool& pool,
                                   std::vector<float>* stage_caps,
                                   std::vector<float>* fused_cap,
                                   std::vector<float>* sky_cap,
                                   std::vector<float>* aux_cap = nullptr);

    // DualDPT auxiliary ray head (opt-in --with-aux GGUF). Runs the independent aux
    // pyramid on the same backbone feats and returns ray (6ch, identity activation)
    // and ray_conf (1ch, expp1) at the aux head resolution (8*pw x 8*ph). Outputs are
    // row-major: ray flat index = (h*Wa + w)*6 + c; ray_conf = h*Wa + w.
    // Returns false if the GGUF has no aux tensors. ray_h/ray_w receive the aux res.
    bool rays(const std::vector<std::vector<float>>& feats, int H, int W,
              std::vector<float>& ray_out, std::vector<float>& ray_conf_out,
              int& ray_h, int& ray_w);

private:
    bool run(const std::vector<std::vector<float>>& feats, int H, int W,
             std::vector<float>& depth_out, std::vector<float>& conf_out,
             std::vector<std::vector<float>>* stages, std::vector<float>* fused,
             std::vector<float>* sky_out = nullptr);
    ModelLoader& ml_;
    Backend& be_;
};

} // namespace da
