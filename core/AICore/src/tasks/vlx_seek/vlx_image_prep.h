// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Bit-exact host image preprocessing for the VLX-Seek C++ chain.
//
// Every function here reproduces ONE upstream python primitive exactly
// (gated byte-for-byte by scripts/compare_image_prep.py):
//
//   decode_image        PIL Image.open(...).convert("RGB")   (libjpeg-turbo
//   ISLOW
//                       + fancy upsampling = PIL defaults; stb fallback for
//                       non-JPEG / builds without libjpeg)
//   resize_u8 kPil      PIL Image.resize (8bpc, 22-bit fixed-point coeffs)
//   resize_u8 kTorch    torchvision resize(uint8, antialias=True) on CPU, i.e.
//                       torch's uint8 AA kernel (int16 coeffs, adaptive
//                       precision)
//   resize_f32_bilinear_aa
//                       F.interpolate(float, bilinear, align_corners=False,
//                       antialias=True) on CPU
//   bf16_round          tensor.to(torch.bfloat16) (round-to-nearest-even)
//   letterbox           WeDetect generate_proposal.letterbox
//   smart_resize        transformers qwen2_vl smart_resize
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vlx {

struct ImageU8 {
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;  // HWC, 3 channels
};

enum class ResampleFilter { Bilinear, Bicubic };
enum class ResampleImpl { kPil, kTorch };

bool decode_image(const std::string& path, ImageU8* out, std::string* err);

// Separable 2-pass AA resample (horizontal first, uint8 intermediate). A pass
// whose size is unchanged is skipped, as in both upstream kernels.
ImageU8 resize_u8(const ImageU8& src,
                  int out_w,
                  int out_h,
                  ResampleFilter f,
                  ResampleImpl impl);

// (C, H, W) float planes -> (C, oh, ow); horizontal pass first, float
// intermediate, float32 coefficients (torch's scalar_t = float path).
std::vector<float> resize_f32_bilinear_aa(
        const std::vector<float>& chw, int C, int H, int W, int oh, int ow);

float bf16_round(float v);

struct Letterbox {
    ImageU8 canvas;  // new_size x new_size, 114 padding
    double ratio = 1.0;
    float off_x = 0.f, off_y = 0.f;  // dw/2, dh/2 (python float division)
};
Letterbox letterbox(const ImageU8& img, int new_size);

void smart_resize(int height,
                  int width,
                  int factor,
                  double min_pixels,
                  double max_pixels,
                  int* h_bar,
                  int* w_bar);

}  // namespace vlx
