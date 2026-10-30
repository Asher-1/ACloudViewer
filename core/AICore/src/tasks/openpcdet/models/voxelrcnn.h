#pragma once

#include "../model.h"
#include "anchor_head.h"
#include "bev_backbone.h"
#include "caddn.h"
#include "point_mlp.h"
#include "sparse_backbone.h"
#include "voxel_backbone_runner.h"

namespace openpcdet {

// AICore port note: upstream declares the Focal VoxelBackBone8x in
// models/focal_backbone.{h,cpp}. Only VoxelRCNN consumes it (focal_former3d
// has no published GGUF assets and is not ported), so its declarations live
// here and its implementation in voxelrcnn.cpp.

// A FocalSparseConv's learned weights: the main conv (with BN1 folded) and the
// bias-free importance conv `conv_imp` (kernel_volume output channels).
struct focal_conv_weights {
    sparse_layer_weights conv;      // [K*Cin, Cout] + bias (BN folded)
    sparse_layer_weights conv_imp;  // [K*Cin, kernel_volume], bias null
    // BN gain (scale) for the multimodal fusion conv only. Py applies BN1 AFTER
    // the image sum-fuse, so the sampled image must be pre-scaled by this gain.
    ggml_tensor * conv_bn_scale = nullptr;  // [Cout]
};

struct focal_backbone_weights {
    sparse_layer_weights conv_input;   // input -> 16
    sparse_layer_weights conv1_0;      // 16 -> 16
    focal_conv_weights conv1_1;        // focal 16 -> 16, voxel_stride 1
    std::vector<sparse_layer_weights> conv2_pre;  // conv2.0/1/2 16->32
    focal_conv_weights conv2_3;        // focal 32 -> 32, voxel_stride 2
    std::vector<sparse_layer_weights> conv3_pre;  // conv3.0/1/2 32->64
    focal_conv_weights conv3_3;        // focal 64 -> 64, voxel_stride 4
    std::vector<sparse_layer_weights> conv4;      // conv4.0/1/2 64->64
    sparse_layer_weights conv_out;     // 64 -> 128
    focal_conv_weights conv_focal_multimodal;     // 16 -> 16, use_img, stride 1
};

bool load_focal_backbone_weights(model_file & file, const hparams & hp,
                                 focal_backbone_weights & weights,
                                 std::string & error);

struct focal_backbone_result {
    // HeightCompression output [height*width*num_bev_features] for the dense head.
    std::vector<float> spatial_features;
    int width = 0;
    int height = 0;
    // RoI multi-scale sources, ordered x_conv2 / x_conv3 / x_conv4.
    std::vector<ext::sparse_indice_map> maps;             // 3 maps
    std::vector<std::vector<float>> layer_features;       // 3 feature sets
};

// Run the Focal VoxelRCNN sparse backbone.
//
//   points / point_count / point_dim : LiDAR point cloud (x,y,z,intensity...)
//   image_features : [16, image_height, image_width] semseg features driving
//                    the hp.focal_multimodal fusion conv, whose weights
//                    consume concatenated image+voxel features and can only
//                    fire when a calibrated image is supplied. A null pointer
//                    skips `conv_focal_multimodal` (same null-image semantics
//                    as upstream focal_backbone): the chain degrades to
//                    LiDAR-only, which is exactly what the points-only
//                    voxelrcnn::forward entry does. The full multimodal
//                    semantics live in voxelrcnn::forward_scene.
//   lidar_to_image / camera_intrinsics : 4x4 calibration matrices used to
//                    project voxel centres onto the image plane for the
//                    multimodal fusion (same semantics as CaDDN). Required
//                    whenever image_features is non-null.
bool run_focal_backbone(engine_backend & backend, const hparams & hp,
                        const focal_backbone_weights & weights,
                        const float * points, int32_t point_count, int32_t point_dim,
                        const std::vector<float> * image_features,
                        int image_width, int image_height,
                        const std::vector<float> & lidar_to_image,
                        const std::vector<float> & camera_intrinsics,
                        focal_backbone_result & result,
                        const tap_sink & sink, std::string & error);

struct voxelrcnn_pool_weights {
    point_mlp_weights feature_in;
    point_mlp_weights position;
    point_mlp_weights output;
};

// Semantic-image branch (Focal VoxelRCNN `semseg` = ResNet50 conv1+layer1 +
// a 1x1 channel-reduce block). Present only for the multimodal focal variant;
// the plain VoxelRCNN has no image branch and `semseg_enabled == false`.
struct voxelrcnn_semseg_weights {
    bool enabled = false;
    conv_bn_block_weights image_stem;                      // conv1 (7x7 s2) + bn1
    std::vector<caddn_bottleneck_weights> image_layers;    // ResNet50 layer1 (3 blocks)
    conv_bn_block_weights reduce;                          // 1x1 conv 256->16 + bn
};

struct voxelrcnn final : model {
    sparse_backbone_weights sparse_w;
    focal_backbone_weights focal_w;      // Focal VoxelRCNN (multimodal) backbone
    bev_backbone_weights backbone_w;
    anchor_head_weights head_w;
    std::vector<voxelrcnn_pool_weights> roi_pool;
    point_mlp_weights roi_shared;
    point_mlp_weights roi_cls_hidden;
    point_mlp_weights roi_cls_output;
    point_mlp_weights roi_reg_hidden;
    point_mlp_weights roi_reg_output;
    voxelrcnn_semseg_weights semseg;

    ggml_tensor * weight(const std::string & name) const { return resolved(name.c_str()); }

    bool load(const std::string & path, const std::string & device, int threads) override;
    bool forward(const float * points, int32_t count, int32_t dimension,
                 std::vector<detection> & detections,
                 const tap_sink & sink = {}) override;
    // Image-capable focal variant: runs the ResNet50 semantic branch and
    // projects the 16-channel image features onto the BEV voxels. The plain
    // LiDAR-only VoxelRCNN falls back to model::forward_scene (which ignores
    // the image and calls forward()).
    bool forward_scene(const scene_input & scene, std::vector<detection> & detections,
                       const tap_sink & sink = {}) override;

private:
    // Shared proposal + RoI refinement + NMS stage, driven by either the plain
    // VoxelBackBone8x (forward) or the Focal VoxelBackBone8xFocal (forward_scene).
    // `sources_prepared` is true when voxel.maps already holds the 3 RoI sources
    // in order (x_conv2/x_conv3/x_conv4) as produced by the focal backbone; false
    // for the plain backbone whose full layer list is indexed by name.
    bool run_roi_heads(const voxel_backbone_result & voxel, bool sources_prepared,
                       std::vector<detection> & detections, const tap_sink & sink);
};

} // namespace openpcdet
