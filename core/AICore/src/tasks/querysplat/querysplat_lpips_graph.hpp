// LPIPS-vgg v0.1 (the `lpips` package, net="vgg") as a ggml graph — the
// learned perceptual metric of evaluations/metrics.py, so the C++ evaluator
// scores without torch.
//
//   lpips(in0, in1, normalize=True):
//     x  = (2*img - 1 - shift) / scale         (ScalingLayer, host side)
//     f_l = VGG16 features relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
//     n_l = f_l / (sqrt(sum_c f_l^2) + 1e-10)   (normalize_tensor)
//     d   = sum_l mean_hw( lin_l . (n_l(in0) - n_l(in1))^2 )   (1x1, no bias)
//
// Weights: models/gguf/lpips-vgg-f32.gguf (scripts/convert_lpips_to_gguf.py:
// torchvision vgg16-397923af.pth features + lpips weights/v0.1/vgg.pth).
// Graph layout: images {W, H, 3, 2} (in0 then in1 of one view pair) — one
// pair per compute keeps the 3x3-im2col working set at ~0.3 GB for 256x256
// beside the resident encoder/decoder runtimes; convolutions are F32 im2col +
// F32 GEMM (ggml_conv_2d would round im2col to F16).
#pragma once

#include "map_anything_backend.hpp"

#include <memory>
#include <string>
#include <vector>

namespace qs {

class LpipsVgg {
   public:
    LpipsVgg(const std::string& gguf_path, map_anything::Backend backend);
    ~LpipsVgg();
    bool valid() const;

    // Per-view lpips(target, prediction, normalize=True) (the official
    // argument order) over (V,3,H,W) [0,1] images.
    bool score(const float* prediction, const float* target, int V, int H, int W,
               std::vector<float>* out);

    // TTO (t3): lpips(target, prediction, normalize=False) — the exact call
    // QuerySplat._tto_losses makes, i.e. NO 2x-1 in front of the scaling
    // layer — together with dLPIPS/dprediction over (V,3,H,W) [0,1] images.
    // The forward runs host-side (same math as the ggml graph, different fp
    // summation order: expect ~1e-6 value differences against score()); the
    // backward ports convTranspose (flipped kernel), ReLU mask, 2x2 maxpool
    // argmax routing and the normalize_tensor unit-norm VJP.
    bool value_and_grad(const float* prediction, const float* target, int V,
                        int H, int W, std::vector<float>* out,
                        std::vector<float>* grad_prediction);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qs
