// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

struct ggml_backend;
typedef struct ggml_backend *ggml_backend_t;

namespace lingbot {

// U-Net sky segmentation converted from the official skyseg.onnx (see
// cpp_ggml/scripts/convert_skyseg.py). Runs the full network as a native
// ggml graph on any backend, replacing the onnxruntime dependency of the
// official viewer's --mask_sky with the same numerics (bilinear resize with
// the pytorch_half_pixel mapping, ImageNet normalization, sigmoid output).
class skyseg {
public:
    skyseg();
    ~skyseg();
    skyseg(const skyseg &) = delete;
    skyseg &operator=(const skyseg &) = delete;

    // Load a GGUF produced by convert_skyseg.py. `backend` must outlive this
    // object; CPU, CUDA and Vulkan all work.
    bool load(const std::string &gguf_path, ggml_backend_t backend);
    void release();

    // Run one image. `rgb_hwc` is interleaved RGB, float, [0,1], shape
    // (height, width, 3) — exactly what the CLI reads from its preprocessed
    // frame buffers after HWC conversion. Returns the sky probability map
    // (320*320 floats, 1 = sky) built by the network's sigmoid head.
    bool infer(const float *rgb_hwc,
               int height,
               int width,
               std::vector<float> &sky_map);

    const std::string &error() const { return error_; }

private:
    bool build_graph();

    struct impl;
    impl *p_ = nullptr;
    std::string error_;
};

// Brings the 320x320 u8 mask lattice back to the frame size with the same
// bit-exact cv2 u8 INTER_LINEAR semantics as the official postprocess
// (segment_sky_from_array: cv2.resize(result_map, (W, H), INTER_LINEAR)).
void skyseg_resize_mask_u8(const unsigned char *src,
                           int sw,
                           int sh,
                           unsigned char *dst,
                           int dw,
                           int dh);

}  // namespace lingbot
