// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "querysplat_tto_loss.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#ifdef QS_USE_OPENMP
#include <omp.h>
#endif

namespace qs {

namespace {

constexpr int kK = 11;        // fused-ssim kernel size
constexpr int kPad = 5;       // "same" padding
constexpr float kC1 = 1e-4f;  // (0.01)^2
constexpr float kC2 = 9e-4f;  // (0.03)^2

// fused-ssim's create_gauss_kernel: 1D gaussian (sigma 1.5) normalized, then
// the outer product. Computed in f32 exactly like the reference stub.
void gauss_kernel1d(float w[kK]) {
    float sum = 0.f;
    for (int k = 0; k < kK; k++) {
        const float x = (float)(k - kK / 2);
        w[k] = std::exp(-(x * x) / (2.f * 1.5f * 1.5f));
        sum += w[k];
    }
    for (int k = 0; k < kK; k++) w[k] /= sum;
}

// Zero-padded "same" 11x11 separable convolution along both axes of one
// (H,W) plane. Pass 1 caches the horizontal pass; the kernel is symmetric
// so the backward uses the same routine (conv with a symmetric kernel is
// its own transpose).
void conv11_separable(
        const float* src, float* dst, int H, int W, const float* w1d) {
    std::vector<float> tmp((size_t)H * W);
#ifdef QS_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float acc = 0.f;
            for (int k = 0; k < kK; k++) {
                const int xx = x + k - kPad;
                if (xx < 0 || xx >= W) continue;  // zero padding
                acc += src[(size_t)y * W + xx] * w1d[k];
            }
            tmp[(size_t)y * W + x] = acc;
        }
    }
#ifdef QS_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float acc = 0.f;
            for (int k = 0; k < kK; k++) {
                const int yy = y + k - kPad;
                if (yy < 0 || yy >= H) continue;
                acc += tmp[(size_t)yy * W + x] * w1d[k];
            }
            dst[(size_t)y * W + x] = acc;
        }
    }
}

// Per-channel fused-ssim forward: fills S map and (train) the three
// derivative maps dm_dmu1, dm_dsigma1_sq, dm_dsigma12.
struct SsimFwd {
    std::vector<float> s_map;      // (V,3,H,W)
    std::vector<float> mu1_map;    // (V,3,H,W)
    std::vector<float> mu2_map;    // (V,3,H,W)
    std::vector<float> dm_dmu1;    // (V,3,H,W)
    std::vector<float> dm_dsig1;   // (V,3,H,W)
    std::vector<float> dm_dsig12;  // (V,3,H,W)
};

void ssim_forward(const float* img1,
                  const float* img2,
                  int V,
                  int H,
                  int W,
                  const float* w1d,
                  SsimFwd* out) {
    const size_t plane = (size_t)H * W;
    out->s_map.assign((size_t)V * 3 * plane, 0.f);
    out->mu1_map.assign((size_t)V * 3 * plane, 0.f);
    out->mu2_map.assign((size_t)V * 3 * plane, 0.f);
    out->dm_dmu1.assign((size_t)V * 3 * plane, 0.f);
    out->dm_dsig1.assign((size_t)V * 3 * plane, 0.f);
    out->dm_dsig12.assign((size_t)V * 3 * plane, 0.f);
#ifdef QS_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < V * 3; b++) {
        const float* x = img1 + (size_t)b * plane;
        const float* y = img2 + (size_t)b * plane;
        std::vector<float> x2(plane), y2(plane), xy(plane);
        for (size_t i = 0; i < plane; i++) {
            x2[i] = x[i] * x[i];
            y2[i] = y[i] * y[i];
            xy[i] = x[i] * y[i];
        }
        std::vector<float> mux(plane), muy(plane), mxx(plane), myy(plane),
                mxy(plane);
        conv11_separable(x, mux.data(), H, W, w1d);
        conv11_separable(y, muy.data(), H, W, w1d);
        conv11_separable(x2.data(), mxx.data(), H, W, w1d);
        conv11_separable(y2.data(), myy.data(), H, W, w1d);
        conv11_separable(xy.data(), mxy.data(), H, W, w1d);
        float* S = &out->s_map[(size_t)b * plane];
        float* m1p = &out->mu1_map[(size_t)b * plane];
        float* m2p = &out->mu2_map[(size_t)b * plane];
        float* g_mu1 = &out->dm_dmu1[(size_t)b * plane];
        float* g_s1 = &out->dm_dsig1[(size_t)b * plane];
        float* g_s12 = &out->dm_dsig12[(size_t)b * plane];
        for (size_t i = 0; i < plane; i++) {
            const float mu1 = mux[i], mu2 = muy[i];
            const float s1 = mxx[i] - mu1 * mu1;
            const float s2 = myy[i] - mu2 * mu2;
            const float s12 = mxy[i] - mu1 * mu2;
            const float A1 = 2.f * mu1 * mu2 + kC1;
            const float A2 = 2.f * s12 + kC2;
            const float B1 = mu1 * mu1 + mu2 * mu2 + kC1;
            const float B2 = s1 + s2 + kC2;
            const float Sval = (A1 * A2) / (B1 * B2);
            S[i] = Sval;
            m1p[i] = mu1;
            m2p[i] = mu2;
            // derivative maps for the backward pass (train=True outputs of
            // the reference fusedssim kernel)
            const float dB1 = -(A1 * A2) / (B1 * B1 * B2);
            const float dB2 = -(A1 * A2) / (B1 * B2 * B2);
            const float dA1 = A2 / (B1 * B2);
            const float dA2 = A1 / (B1 * B2);
            g_mu1[i] = 2.f * mu2 * dA1 + 2.f * mu1 * dB1;
            g_s1[i] = dB2;
            g_s12[i] = 2.f * dA2;  // sigma12 only enters through A2
        }
    }
}

}  // namespace

float fused_ssim_value(
        const float* img1, const float* img2, int V, int H, int W) {
    float w1d[kK];
    gauss_kernel1d(w1d);
    SsimFwd f;
    ssim_forward(img1, img2, V, H, W, w1d, &f);
    const size_t n = (size_t)V * 3 * H * W;
    double acc = 0;
    for (size_t i = 0; i < n; i++) acc += (double)f.s_map[i];
    return (float)(acc / (double)n);
}

void fused_ssim_backward(const float* img1,
                         const float* img2,
                         int V,
                         int H,
                         int W,
                         const float* d_ssim_map,
                         float* d_img1) {
    float w1d[kK];
    gauss_kernel1d(w1d);
    SsimFwd f;
    ssim_forward(img1, img2, V, H, W, w1d, &f);
    const size_t plane = (size_t)H * W;
#ifdef QS_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < V * 3; b++) {
        // per-b scratch: the b loop is parallel, so nothing here may be shared
        const float* x = img1 + (size_t)b * plane;
        const float* y = img2 + (size_t)b * plane;
        const float* dS = d_ssim_map + (size_t)b * plane;
        const float* g_mu1 = &f.dm_dmu1[(size_t)b * plane];
        const float* g_s1 = &f.dm_dsig1[(size_t)b * plane];
        const float* g_s12 = &f.dm_dsig12[(size_t)b * plane];
        std::vector<float> term1b(plane), term2b(plane), term3b(plane),
                t1c(plane), t2c(plane), t3c(plane);
        const float* m1 = &f.mu1_map[(size_t)b * plane];
        const float* m2 = &f.mu2_map[(size_t)b * plane];
        for (size_t i = 0; i < plane; i++) {
            // total dL/dmu1: the direct A1/B1 path PLUS the -mu1^2 term of
            // sigma1_sq and the -mu1*mu2 term of sigma12 (fusedssim_backward)
            term1b[i] = dS[i] *
                        (g_mu1[i] - 2.f * m1[i] * g_s1[i] - m2[i] * g_s12[i]);
            // sigma maps: dsigma1_sq = dS*dm_dsigma1_sq (conv(x^2) path),
            // dsigma12 = dS*dm_dsigma12 (conv(x*y) path)
            term2b[i] = dS[i] * g_s1[i];
            term3b[i] = dS[i] * g_s12[i];
        }
        // transpose convolutions (symmetric kernel: same routine). NOTE the
        // pixel-dependent factors sit OUTSIDE the convolutions:
        // dL/dx = convT(dmu1) + 2x * convT(ds1) + y * convT(ds12) — putting
        // 2x/y inside the conv (an easy mistake) breaks the parity by ~100%.
        conv11_separable(term1b.data(), t1c.data(), H, W, w1d);
        conv11_separable(term2b.data(), t2c.data(), H, W, w1d);
        conv11_separable(term3b.data(), t3c.data(), H, W, w1d);
        float* out = d_img1 + (size_t)b * plane;
        for (size_t i = 0; i < plane; i++)
            out[i] = t1c[i] + 2.f * x[i] * t2c[i] + y[i] * t3c[i];
    }
}

bool tto_loss_value_and_grad(const float* pred,
                             const float* gt,
                             int V,
                             int H,
                             int W,
                             float lpips_weight,
                             const float* d_lpips,
                             TtoLossOut* out) {
    const size_t n = (size_t)V * 3 * H * W;
    if (n == 0) return false;
    float w1d[kK];
    gauss_kernel1d(w1d);
    SsimFwd f;
    ssim_forward(pred, gt, V, H, W, w1d, &f);
    double ssim_acc = 0;
    for (size_t i = 0; i < n; i++) ssim_acc += (double)f.s_map[i];
    out->ssim = (float)(ssim_acc / (double)n);
    // L1 mean over all elements (F.l1_loss default 'mean')
    double l1_acc = 0;
    for (size_t i = 0; i < n; i++) l1_acc += (double)std::fabs(pred[i] - gt[i]);
    out->l1 = (float)(l1_acc / (double)n);
    // dL/dpred = sign(pred-gt)/n  +  (-0.1) * dssim/dpred  +  w * d_lpips
    // (d loss_total/d ssim_map = 0.2 * (-1/2) / n per element)
    const float inv_n = 1.0f / (float)n;
    std::vector<float> d_ssim_map(n);
    for (size_t i = 0; i < n; i++) d_ssim_map[i] = -0.1f * inv_n;
    out->d_pred.assign(n, 0.f);
    fused_ssim_backward(pred, gt, V, H, W, d_ssim_map.data(),
                        out->d_pred.data());
    for (size_t i = 0; i < n; i++) {
        const float s = pred[i] >= gt[i] ? 1.0f : -1.0f;
        float g = s * inv_n + out->d_pred[i];
        if (lpips_weight > 0.f && d_lpips) g += lpips_weight * d_lpips[i];
        out->d_pred[i] = g;
    }
    return true;
}

}  // namespace qs
