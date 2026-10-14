// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// PaddleX rec crop (CropByPolys, det_box_type "quad"): the pixels the
// recognizer sees. get_minarea_rect_crop re-fits every det box with
// cv2.minAreaRect + cv2.boxPoints before get_rotate_crop_image warps it with
// cv2.warpPerspective(INTER_CUBIC, BORDER_REPLICATE). Both steps are ported
// from OpenCV 4.10 so the crop is bit-identical; a bilinear warp of the raw
// det quad changed 24% of the official recognizer's ICDAR2015 word texts.
// The same re-fit is DBPostProcess.get_mini_boxes (ocr_db_postprocess.cpp).

#include <cstdint>

#include "common/image_utils.hpp"

namespace aicore {
namespace ocr {

// cv2.minAreaRect of `count` integer points, its cv2.boxPoints, then the
// (left pair, right pair) ordering shared by get_minarea_rect_crop and
// DBPostProcess.get_mini_boxes. `minSide` (optional) receives
// min(rect.size), get_mini_boxes' sside.
void minAreaRectBox(const int (*pts)[2],
                    int count,
                    float box[4][2],
                    float* minSide = nullptr);

// get_rotate_crop_image's target size: int() of the longer float32 edge
// norms (width from edges 0-1 / 2-3, height from 0-3 / 1-2).
void rotateCropSize(const float box[4][2], int* width, int* height);

// cv2.warpPerspective(src, getPerspectiveTransform(box, (0,0)(w,0)(w,h)
// (0,h)), (w, h), flags=INTER_CUBIC, borderMode=BORDER_REPLICATE) into a
// tightly packed w*h*3 destination; output channel c samples
// src.channel_index(c). Returns false (dst zero-filled) when the transform
// is singular, where cv2 would also not produce a usable crop.
bool warpPerspectiveCubicReplicate(const aicore::img::Image& src,
                                   const float box[4][2],
                                   int width,
                                   int height,
                                   std::uint8_t* dst);

}  // namespace ocr
}  // namespace aicore
