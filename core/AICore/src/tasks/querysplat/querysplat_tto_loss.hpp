// Host-side TTO loss: fused_ssim (train=True) forward/backward + L1, the
// exact math of scripts/rendering/fused_ssim.py's FusedSSIMMap CUDA kernels
// and QuerySplat._tto_losses:
//   loss_total = L1 + 0.2 * (1 - ssim)/2 + w * lpips   (lpips wired by t3)
//
// fused_ssim semantics (fused-ssim package, 11x1x11 separable Gaussian
// kernel, sigma 1.5, zero-padded "same" convolution):
//   mu1/mu2 = conv(img1/img2), sigma1_sq = conv(img1^2) - mu1^2,
//   sigma12 = conv(img1*img2) - mu1*mu2,
//   A1 = 2*mu1*mu2 + C1 (C1 = 1e-4), A2 = 2*sigma12 + C2 (C2 = 9e-4),
//   B1 = mu1^2 + mu2^2 + C1, B2 = sigma1_sq + sigma2_sq + C2,
//   S = A1*A2/(B1*B2); loss uses S.mean().
// The backward returns dL/dimg1 only (img2 = ground truth is constant).
#pragma once

#include <vector>

namespace qs {

struct TtoLossOut {
    float l1 = 0.f;
    float ssim = 0.f;  // fused_ssim mean (before the 0.2*(1-s)/2 fold)
    std::vector<float> d_pred;  // (V,3,H,W) dL/dpred for L1 + 0.2*(1-s)/2
};

// pred/gt: (V,3,H,W) torch C-order in [0,1]. d_lpips (may be null when
// lpips_weight == 0): the LPIPS term's dL/dpred, added scaled by
// lpips_weight. Returns false on allocation failure (not expected).
bool tto_loss_value_and_grad(const float* pred, const float* gt, int V,
                             int H, int W, float lpips_weight,
                             const float* d_lpips, TtoLossOut* out);

// fused_ssim forward only (mean over channels and pixels), for tests.
float fused_ssim_value(const float* img1, const float* img2, int V, int H,
                       int W);

// fused_ssim backward: dL/dimg1 given dL/dS-map already folded with the
// loss scaling (d_loss/d_ssim_map, per pixel per channel).
void fused_ssim_backward(const float* img1, const float* img2, int V, int H,
                         int W, const float* d_ssim_map, float* d_img1);

}  // namespace qs
