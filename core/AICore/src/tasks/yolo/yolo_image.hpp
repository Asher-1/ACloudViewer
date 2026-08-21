#pragma once


// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: AGPL-3.0
// ----------------------------------------------------------------------------
//
// Letterbox preprocessing and depth-map restoration. In-tree port of
// ultralytics-ggml cpp_ggml/src/image_io.{hpp,cpp}, reduced to the pure
// algorithms the C API needs: the upstream stb_image load/save, the bitmap
// glyph renderer and the colorized depth PNG writer live on the plugin side
// (Qt QImage / QPainter), matching the qRFDetr split.

#include <vector>

#include "tasks/yolo/yolo_common.hpp"
#include "tasks/yolo/yolo_postprocess.hpp"


namespace yolo {

/* Non-owning view over an interleaved RGB8 image (borrowed from the caller
 * for the duration of the call; the C API never copies it). */
struct Image {
    int w = 0, h = 0;
    const uint8_t* rgb = nullptr;
};

// Ultralytics-equivalent LetterBox(auto=True, stride=32): resize keeping
// aspect then pad to a stride-multiple rectangle inside imgsz x imgsz.
// Bilinear resampling matches cv2.INTER_LINEAR bit-for-bit. `out` receives
// the CHW float canvas [3, canvas_h, canvas_w] in [0, 1]; the canvas dims
// are reported through info.imgsz_w / info.imgsz_h (they vary with the image
// aspect ratio — the session rebuilds its graph when they change).
void letterbox_image(const Image& img, int imgsz, LetterboxInfo& info,
                     std::vector<float>& out);

// Map boxes from the letterboxed canvas back to original image pixels.
void unscale_boxes(std::vector<Detection>& dets, const LetterboxInfo& info);

// Remap instance masks from the letterbox-canvas space back to the source
// image space. compose_masks() emits one canvas-space window per instance
// (SegMask.x/y/w/h); consumers that overlay masks on the original image
// need source-space pixels. Each mask becomes a full-size image_w x image_h
// bitmap (x = y = 0) sampled nearest-neighbor from its canvas window, so
// the result aligns 1:1 with boxes already processed by unscale_boxes().
void unscale_masks(std::vector<SegMask>& masks, const LetterboxInfo& info,
                   int image_w, int image_h);

// Restore a model-resolution depth map to the original image size, matching
// DepthPredictor's bilinear resize, letterbox crop, and final resize.
std::vector<float> restore_depth(const std::vector<float>& depth, int depth_w,
                                 int depth_h, const LetterboxInfo& info,
                                 int image_w, int image_h);

}  // namespace yolo
