// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// GKDT ggml runtime - PIL-compatible geometry and preprocessing (in-tree port
// of upstream cpp_ggml/src/image_io.*).
//
// The official pipeline (test_real_world/gkd_inference_lib) is:
//   crop GT bbox -> PIL-bilinear resize (longer side = 384) -> center-pad with
//   (124,116,104) -> ToTensor/Normalize(ImageNet mean/std)
// The Resize transform resamples with PIL.Image.BILINEAR (antialiased), which
// differs from OpenCV's INTER_LINEAR; we reimplement PIL's triangle filter.
//
// File decoding intentionally lives OUTSIDE the core: the AICore contract
// feeds decoded pixels through a borrowed aicore_image_view (see
// rgb_from_view below); path-based decoding belongs to the compatibility
// wrapper in gkd_capi.cpp.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "aicore/image_view.h"

namespace gkd {

struct RgbImage {
    int w = 0, h = 0;
    std::vector<uint8_t> data;  // RGB, 3*w*h

    RgbImage() = default;
    RgbImage(int w_, int h_) : w(w_), h(h_), data((size_t)w_ * h_ * 3) {}
    uint8_t* row(int y) { return data.data() + (size_t)y * w * 3; }
    const uint8_t* row(int y) const {
        return data.data() + (size_t)y * w * 3;
    }
};

// Copy a borrowed stride-aware view into a tightly-packed RGB staging image
// (the single preprocess-entry conversion; supports every view format).
bool rgb_from_view(const aicore_image_view& view, RgbImage& out);

// Geometry transform record (mirrors the official meta/scale_trans).
struct ScaleTrans {
    float scale = 1.0f;
    float offset_x = 0.0f;
    float offset_y = 0.0f;
};

// PIL.Image.resize(BILINEAR) with the antialiased triangle filter.
RgbImage resize_bilinear_pil(const RgbImage& src, int dst_w, int dst_h);

// torchvision.transforms.functional.pad with a constant fill color.
RgbImage pad_image(const RgbImage& src,
                   int left,
                   int top,
                   int right,
                   int bottom,
                   uint8_t fill[3]);

// Crop [x0, y0, x1] (exclusive x1/y1), mirroring PIL Image.crop ltrb.
RgbImage crop_image(const RgbImage& src, int x0, int y0, int x1, int y1);

// Official ROI preprocessing: crop the bbox (RandomCrop(crop_gt_bbox=True)),
// resize the longer side to `square` (PIL bilinear), center-pad to square.
// Keypoints in `kps_in` (original image coords, N pairs) are transformed like
// the official pipeline; pass kps_in=nullptr when there are none.
struct PreprocessResult {
    RgbImage img;
    ScaleTrans trans;               // (scale, offset_x, offset_y)
    std::vector<float> kps_norm;    // N x 2 in -1..1 (post-transform), optional
    std::vector<uint8_t> kps_valid; // N flags, optional
};

PreprocessResult preprocess_roi(const RgbImage& src,
                                float bbox[4],
                                int square,
                                const float* kps_in,
                                int n_kps,
                                const uint8_t* kps_vis_in);

// Convert an RGB image to the ImageNet-normalized CHW float32 blob (parity
// dumps) and to the patch-embed im2col blob: {IC*KH*KW, L} with
// k = kw + kh*KW + ic*KH*KW, l = w + h*LW (exactly what conv2d k==s produces).
void image_to_chw_normalized(const RgbImage& img,
                             const float mean[3],
                             const float stdv[3],
                             float* out_chw);
void image_to_im2col_normalized(const RgbImage& img,
                                const float mean[3],
                                const float stdv[3],
                                int patch,
                                float* out_im2col);

// Recover normalized -1..1 keypoints to original image coordinates, following
// mytransforms.recover_kps:  (k/2 + 0.5) * L + offset, then / scale.
void recover_kps(const float* kps_norm /*N x 2*/,
                 int n,
                 int square_len,
                 const ScaleTrans& trans,
                 float* out /*N x 2*/);

}  // namespace gkd
