#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include <vector>

namespace aicore {
namespace depth {

// GSDPT (3D-Gaussian) head (DA3 giant). Subclasses the single-head DPT main
// path (same fusion pyramid as DptHead, but norm_type="idt" -> NO LayerNorm),
// then injects the input image through `images_merger` and emits 38 channels via
// output_conv2. activate_head_gs splits into xyz (37, linear) and conf (1,
// sigmoid), channels-last. Mirrors GSDPT._forward_impl.
class GsHead {
public:
    GsHead(ModelLoader& ml, Backend& be) : ml_(ml), be_(be) {}

    // feats     : 4 giant out-layer features, each [N*C] = [256*3072]
    //             (flat index = token*C + channel).
    // image_chw : the merger input image [3*H*W], CHW (c-major then h then w),
    //             i.e. the same ImageNet-NORMALIZED tensor the network forward
    //             feeds to the gs_head (NOT a [0,1] image).
    // raw_gs    : [H*W*37], channels-LAST: (h,w,c) at (h*W+w)*37 + c.
    // gs_conf   : [H*W], row-major (h,w; w fastest).
    bool raw_gaussians(const std::vector<std::vector<float>>& feats,
                       const std::vector<float>& image_chw, int H, int W,
                       std::vector<float>& raw_gs, std::vector<float>& gs_conf);

private:
    ModelLoader& ml_;
    Backend& be_;
};

} // namespace depth
} // namespace aicore
