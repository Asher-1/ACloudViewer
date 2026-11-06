// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// HFRE: Hybrid Fine-Grained Region Encoder, host-side C++ port.
//
// Port of hybrid_finegrained_region_encoder.py for the released 10B config:
//   mm_use_vision_tower_object_feature=True, mm_use_simpleFPN_for_vt=True,
//   combination="concat", roi_output_size=14, pos dim = 6656 (= 4608 + 2048).
//
//   aux path : 4 RADIO levels (same 64x64 grid) -> channel concat (4608)
//              -> RoIAlign 14x14 (scale 1/16) -> mean pool        (N, 4608)
//   vt path  : main-tower merged feature -> SimpleFPN 4 levels x 512
//              -> per-level RoIAlign -> mean -> concat            (N, 2048)
//   fusion   : concat[aux, vt]                                     (N, 6656)
//   position : sine embedding of normalized cxcywh (from vt-path boxes)
//              added to the concat vector                          (+6656)
//
// The aux branch has NO learned parameters — parity against
// torchvision.ops.roi_align alone validates it. The SimpleFPN conv tower
// itself runs inside the P1 mmproj graph; HFRE consumes its 4 levels.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "tasks/vlx_seek/vlx_types.h"

namespace vlx {

// One bilinear sampling tap of a RoIAlign grid point (torchvision
// pre_calc_for_bilinear_interpolate semantics: edge-snapped 4-neighbor
// interpolation). Exported so the GPU RoIAlign path can precompute the
// gather indices/weights with the exact same sampling geometry.
struct PreCalc {
    int pos1, pos2, pos3, pos4;
    float w1, w2, w3, w4;
};

// Line-for-line port of torchvision 0.25 cpu/roi_align_common.h
// pre_calc_for_bilinear_interpolate (fills bin-major: ph, pw, grid_y,
// grid_x; zero-weight taps when a sample falls fully outside).
void pre_calc_bilinear(int height,
                       int width,
                       int pooled,
                       float roi_start_h,
                       float roi_start_w,
                       float bin_h,
                       float bin_w,
                       int grid_h,
                       int grid_w,
                       std::vector<PreCalc>* out);

class GgufFile;

class HfreEncoder {
public:
    explicit HfreEncoder(const HfreConfig& cfg);

    // aux-only encode (validated directly against torchvision; the vt branch
    // needs the P1 main-tower tap and is appended through encode_vt()).
    bool encode_aux(const std::vector<Box>& boxes,
                    const std::vector<FeatureLevel>& aux_levels,
                    std::vector<float>* out_aux_nxc,
                    std::string* err = nullptr);

    // vt branch: consumes the 4 SimpleFPN levels produced by the mmproj graph.
    // levels are ordered as Python's simple_fpn output (level 0 first).
    // feature_strides must match the FPN pyramid (level 0 = finest).
    bool encode_vt(const std::vector<Box>& boxes,
                   const std::vector<FeatureLevel>& vt_fpn_levels,
                   const std::vector<float>& feature_strides,
                   std::vector<float>* out_vt_nxc,
                   std::string* err = nullptr);

    // Full fusion: concat(aux, vt) + sine pos embed of the boxes in
    // ORIGINAL-image pixels. Returns (N, final_dim) row-major.
    bool fuse(const std::vector<float>& aux_nxc,
              const std::vector<float>& vt_nxc,
              const std::vector<Box>& boxes,
              int aux_channels,
              int vt_channels,
              const std::vector<FeatureLevel>& vt_grid_ref,
              std::vector<float>* out_object_features,
              std::string* err = nullptr);

    // Torchvision-parity RoIAlign on a single NCHW level.
    // boxes are in ORIGINAL image pixels; scaled by spatial_scale inside.
    static void roi_align_mean_pool(const FeatureLevel& level,
                                    const std::vector<Box>& boxes_px,
                                    float spatial_scale,
                                    int out_size,
                                    bool aligned,
                                    std::vector<float>* out_nxc,
                                    int n_threads = 1);

private:
    // sine embedding: input (N,4) normalized cxcywh -> (N, pos_embedding_dim)
    // (block order y, x, w, h; each block pos_embedding_dim/4 wide).
    void sine_embed(const std::vector<std::array<float, 4>>& cxcywh_norm,
                    std::vector<float>* out_nxd) const;

    HfreConfig cfg_;
};

}  // namespace vlx
