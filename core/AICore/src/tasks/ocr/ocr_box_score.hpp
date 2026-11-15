// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>

namespace aicore {
namespace ocr {

// PaddleX DBPostProcess.box_score_fast: the mean of `prob` (w x h, row-major)
// over the pixels cv2.fillPoly sets for `quad` once its corners are shifted
// by the clamped floor/ceil bounding box in float32 and truncated to int32.
// The rasterized set includes the Bresenham-drawn polygon edges, so it is not
// "pixel centers inside the rectangle"; that geometric rule scores thin text
// higher (measured +0.058 mean on the same map) and admits extra boxes at the
// 0.6 gate. Returns double like cv2.mean so the gate compares in the same
// precision as upstream.
double dbBoxScoreFast(const float* prob, int w, int h, const float quad[4][2]);

// cv::fillPoly(mask, {pts}, 1, LINE_8, shift 0) on an mw x mh uint8 mask:
// LINE_8 edges (cv::LineIterator with clipLine) plus FillEdgeCollection's
// fixed-point scanline interior. Exposed for the rasterization parity test.
void fillPolyLine8(const int (*pts)[2],
                   int count,
                   std::uint8_t* mask,
                   int mw,
                   int mh);

}  // namespace ocr
}  // namespace aicore
