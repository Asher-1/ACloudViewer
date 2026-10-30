#pragma once

// BEVFusion — multi-modal 3-D detector (pcdet.models.detectors.bevfusion).
//
// Forward orchestration:
//   1. LiDAR branch: voxelize → VoxelResBackBone8x → HeightCompression
//   2. Camera branch: SwinTransformer → GeneralizedLSSFPN → DepthLSSTransform
//   3. Fusion: ConvFuser(lidar_bev, img_bev) → fused BEV
//   4. Head: BaseBEVBackbone → TransFusionHead → decode + NMS
//
// Inherits from transfusion to reuse the TransFusion head logic
// (forward_spatial) and weight loading for sparse backbone + head.

#include "transfusion.h"
#include "swin_transformer.h"
#include "lss_fpn.h"
#include "depth_lss.h"
#include "conv_fuser.h"

#include <string>
#include <vector>

namespace openpcdet {

struct bevfusion : transfusion {
    // Camera branch weights
    swin_weights        swin_w;
    lss_fpn_weights     fpn_w;
    depth_lss_weights   lss_w;
    conv_fuser_weights  fuser_w;

    bool load(const std::string & gguf_path, const std::string & device, int n_threads) override;
    bool forward(const float * points, int32_t n_points, int32_t point_dim,
                 std::vector<detection> & out_dets,
                 const tap_sink & sink = {}) override;
    bool forward_scene(const scene_input & scene,
                       std::vector<detection> & out_dets,
                       const tap_sink & sink = {}) override;
};

} // namespace openpcdet
