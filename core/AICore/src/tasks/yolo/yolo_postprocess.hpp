#pragma once


// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: AGPL-3.0
// ----------------------------------------------------------------------------
//
// Detection decoding for the raw detect head output and segment mask
// composition. In-tree port of ultralytics-ggml cpp_ggml/src/postprocess.{hpp,cpp}.

#include <vector>

#include "tasks/yolo/yolo_common.hpp"


namespace yolo {

struct PostprocConfig {
    float conf_thres = 0.25f;  // predict conf from cfg/default.yaml
    float iou_thres = 0.7f;    // predict iou from cfg/default.yaml
    int max_det = 300;
    int max_nms = 30000;
};

/* Decode the raw detect output into boxes in letterbox-canvas coordinates.
 *
 * raw is [no, na] with element (c, a) at raw[c * na + a]; no = 4*reg_max + nc
 * (+ nm mask-coefficient rows for segment models). anchors is [na*2] holding
 * (x+0.5, y+0.5) per anchor (unscaled), strides is [na]. Mirrors ultralytics
 * Detect._inference + non_max_suppression:
 *  - v8 heads: per-anchor max sigmoid class + conf filter, DFL softmax over
 *    reg_max, dist2bbox(xywh) * stride, xywh->xyxy, greedy class-aware NMS,
 *    top max_det (score descending, ties by anchor index).
 *  - end2end heads (yolo26): dist2bbox(xyxy) * stride, per-anchor max
 *    sigmoid class + conf filter, top max_det by score. No NMS.
 * Each Detection carries its anchor index so segment masks can pick up the
 * matching mask coefficients.
 */
std::vector<Detection> postprocess(const std::vector<float>& raw, int no,
                                   int na, const ModelMeta& meta,
                                   const float* anchors, const float* strides,
                                   const PostprocConfig& cfg);

/* One binary instance mask per detection, cropped to its letterbox-canvas box.
 *
 * Mirrors ultralytics process_mask: mask = sigmoid(mc[nm] @ proto) crop-masked
 * to the box on the H/4 proto grid, then nearest-upsampled to the canvas. With
 * a 640/160 canvas the nearest 4x upsample is exactly one proto cell per 4x4
 * canvas block, so the composition collapses to a per-canvas-pixel lookup
 * (px, py) = (x * pw / cw, y * ph / ch) — no intermediate mask tensor.
 * detections must still be in canvas coordinates (call before unscale_boxes).
 */
struct SegMask {
    std::vector<uint8_t> bits;  // [w * h], 1 = inside the instance
    int x = 0, y = 0, w = 0, h = 0;  // canvas-space window
};

std::vector<SegMask> compose_masks(const std::vector<Detection>& dets,
                                   const std::vector<float>& raw, int na,
                                   const ModelMeta& meta,
                                   const std::vector<float>& proto,
                                   int proto_w, int proto_h, int canvas_w,
                                   int canvas_h);

}  // namespace yolo
