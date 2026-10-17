// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// PaddleX DBPostProcess.boxes_from_bitmap (box_type "quad", score_mode
// "fast", no dilation) followed by the OCR pipeline's SortQuadBoxes: the det
// boxes the official pipeline reports and crops. Every step is a port of the
// code upstream runs (OpenCV 4.10 findContours RETR_LIST/CHAIN_APPROX_SIMPLE,
// minAreaRect/boxPoints, contourArea/arcLength, pyclipper 1.4.0 / Clipper
// 6.4.2 JT_ROUND offset) so the integer boxes are identical for the same
// probability map. The historical flood fill + float min-area rectangle +
// approximate unclip differed from upstream on 46 of 487 ICDAR2015 boxes
// fed the SAME map, and dropped the hole contours RETR_LIST also returns.

#include <array>
#include <cstdint>
#include <vector>

namespace aicore {
namespace ocr {

struct DbBox {
    int quad[4][2];  // dt_polys corners (int16 range), image pixels
    double score;    // box_score_fast of the pre-unclip box
};

struct DbParams {
    float thresh = 0.3f;     // pred > thresh (float32 compare, like numpy)
    double boxThresh = 0.6;  // box_thresh > score drops the box
    double unclipRatio = 1.5;
    int maxCandidates = 1000;
    int minSize = 3;
};

// `pred` is the w x h probability map; destW/destH the source image size the
// boxes are scaled to. `scratch` is caller-owned and reused across frames.
std::vector<DbBox> dbBoxesFromBitmap(const float* pred,
                                     int w,
                                     int h,
                                     int destW,
                                     int destH,
                                     const DbParams& params,
                                     std::vector<std::int8_t>& scratch);

// cv2.findContours(bitmap, RETR_LIST, CHAIN_APPROX_SIMPLE) of a 0/1 w x h
// bitmap, in OpenCV's output order. Exposed for the parity test.
std::vector<std::vector<std::array<int, 2>>> findContoursList(
        const std::uint8_t* bitmap,
        int w,
        int h,
        std::vector<std::int8_t>& scratch);

// pyclipper.PyclipperOffset().AddPath(box, JT_ROUND, ET_CLOSEDPOLYGON) then
// Execute(distance): the single output path of a convex box, in pyclipper's
// point order (box corners are truncated to integers like pyclipper does).
std::vector<std::array<long long, 2>> clipperOffsetRound(const float box[4][2],
                                                         double distance);

// paddlex SortQuadBoxes: stable sort by (top-left y, x), then the 10 px
// same-line bubble pass.
void sortQuadBoxes(std::vector<DbBox>& boxes);

}  // namespace ocr
}  // namespace aicore
