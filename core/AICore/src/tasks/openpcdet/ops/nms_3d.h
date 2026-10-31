// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// 3D rotated IoU + NMS, ported from pcdet/ops/iou3d_nms/src/iou3d_cpu.cpp with
// the torch dependency stripped. Used in the post-processing stage.
//
// Boxes are [x, y, z, dx, dy, dz, heading] (7-D) — same layout as OpenPCDet's
// `pred_boxes`. The IoU is computed on the BEV projection (x,y,dx,dy,heading),
// matching OpenPCDet's `boxes_iou3d_gpu` semantics for the NMS step.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace openpcdet {

// IoU on BEV between two 7-D boxes. Pure CPU, no torch.
float iou_bev(const float* box_a, const float* box_b);

// Class-agnostic 3D NMS. boxes: [N, 7] row-major. scores: [N]. Returns the
// kept indices in descending score order, after suppressing overlaps >
// nms_thresh. `max_boxes` caps the output (after NMS, top-k by score).
std::vector<int32_t> nms_3d(const std::vector<float>& boxes,
                            const std::vector<float>& scores,
                            float nms_thresh,
                            int max_boxes);

}  // namespace openpcdet
