// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// GGUF model file for OpenPCDet architectures: metadata, hyperparameters,
// tensor access. Written by scripts/convert_<arch>.py.
//
// Multiple architectures share this loader: hparams is a tagged union keyed on
// `general.architecture`. Today: "openpcdet.pointpillar". Adding a new
// architecture means adding a new arch enum + a parse branch in
// gguf_loader.cpp and a forward() in models/<name>.cpp.
#pragma once

#include <ggml.h>
#include <gguf.h>

#include <string>
#include <vector>

namespace openpcdet {

enum class arch_kind {
    unknown = 0,
    pointpillar,
    second,
    voxel_detector,
    voxelnext,
    pointrcnn,
    pvrcnn,
    voxelrcnn,
    transfusion,
    parta2,
    bevfusion,
    caddn,
    focal_former3d,
    fastpillars,
};

enum class dense_head_kind {
    anchor_single = 0,
    anchor_multi = 1,
    center = 2,
    voxelnext = 3,
    transfusion = 4,
};

// PointPillar hyperparameters. Lives in the GGUF as
// "openpcdet.pointpillar.*" keys. Other architectures get their own sub-struct
// under the same `hparams` umbrella (kept as a flat struct for now — split
// when a second architecture lands).
struct hparams {
    arch_kind arch = arch_kind::unknown;

    // --- dataset / voxel grid ---
    std::vector<float> point_cloud_range;  // [x0,y0,z0,x1,y1,z1]
    std::vector<float> voxel_size;         // [vx,vy,vz]
    std::vector<int32_t> grid_size;        // [nx,ny,nz]  (nx along x)
    int32_t max_points_per_voxel = 32;
    int32_t max_voxels = 16000;

    // --- point features ---
    int32_t num_point_features = 4;  // x,y,z,intensity for kitti
    bool use_absolute_xyz = true;
    bool with_distance = false;

    // --- VFE (PillarVFE) ---
    std::vector<int32_t> vfe_num_filters;  // e.g. [64]
    bool vfe_use_norm = true;
    bool vfe_dynamic = false;
    bool sparse_residual = false;
    std::vector<int32_t> sparse_channels;
    std::vector<int32_t> sparse_kernel_sizes;
    int32_t sparse_out_channels = 128;

    // --- map_to_bev (PointPillarScatter) ---
    int32_t num_bev_features = 64;

    // --- backbone_2d (BaseBEVBackbone) ---
    std::vector<int32_t> bb_layer_nums;  // per level
    std::vector<int32_t> bb_layer_strides;
    std::vector<int32_t> bb_num_filters;
    std::vector<float> bb_upsample_strides;
    std::vector<int32_t> bb_num_upsample_filters;
    std::vector<int32_t> bb_deblock_transpose;

    // --- dense_head (AnchorHeadSingle) ---
    int32_t num_class = 3;
    dense_head_kind head_kind = dense_head_kind::anchor_single;
    int32_t num_dir_bins = 2;
    bool use_direction_classifier = true;
    int32_t code_size = 7;
    // anchor generator: per-anchor-set config
    std::vector<std::vector<std::vector<float>>>
            anchor_sizes;                              // [set][size_idx][3]
    std::vector<std::vector<float>> anchor_rotations;  // [set][rot_idx]
    std::vector<std::vector<float>> anchor_bottom_heights;  // [set][h_idx]
    std::vector<int32_t> feature_map_strides;
    std::vector<int32_t> anchor_align_center;  // [set], 0 or 1
    float dir_offset = 0.0f;
    float dir_limit_offset = 0.0f;
    bool encode_angle_by_sincos = false;
    bool multi_class_nms = false;

    // AnchorHeadMulti metadata. All flattened arrays are grouped by head.
    int32_t multi_num_heads = 0;
    int32_t multi_shared_channels = 0;
    bool multi_separate_reg = false;
    int32_t multi_middle_convs = 0;
    int32_t multi_middle_channels = 0;
    std::vector<int32_t> multi_head_class_counts;
    std::vector<int32_t> multi_head_anchor_counts;
    std::vector<int32_t> multi_label_indices;
    std::vector<int32_t> multi_anchor_set_indices;
    std::vector<int32_t> multi_reg_channels;

    // CenterHead metadata. Branches always contain center, center_z, dim,
    // rotation and heatmap; velocity is optional.
    int32_t center_num_heads = 0;
    int32_t center_shared_channels = 0;
    int32_t center_middle_convs = 0;
    int32_t center_feature_map_stride = 1;
    int32_t center_max_objects = 500;
    bool center_has_velocity = false;
    bool center_has_iou = false;
    int32_t center_head_kernel_size = 3;
    bool center_double_flip = false;
    std::vector<int32_t> center_class_counts;
    std::vector<int32_t> center_label_indices;
    std::vector<float> center_post_limit_range;
    std::vector<float> center_rectifier;
    std::vector<float> center_nms_thresholds;
    std::vector<int32_t> center_nms_pre_maxsizes;
    std::vector<int32_t> center_nms_post_maxsizes;

    int32_t transfusion_num_proposals = 0;
    int32_t transfusion_hidden_channels = 0;
    int32_t transfusion_num_heads = 0;
    int32_t transfusion_ffn_channels = 0;
    int32_t transfusion_nms_kernel_size = 0;
    int32_t transfusion_feature_map_stride = 0;
    int32_t transfusion_middle_channels = 0;
    bool transfusion_has_velocity = false;
    std::vector<float> transfusion_post_center_range;

    // Multi-camera image branch shared by BEVFusion-style scene graphs.
    int32_t image_num_cameras = 0;
    std::vector<int32_t> image_input_size;
    std::vector<int32_t> image_feature_size;
    int32_t image_swin_embed_dims = 0;
    std::vector<int32_t> image_swin_depths;
    std::vector<int32_t> image_swin_num_heads;
    int32_t image_swin_window_size = 0;
    int32_t image_swin_mlp_ratio = 0;
    std::vector<int32_t> image_swin_out_indices;
    std::vector<int32_t> image_neck_in_channels;
    int32_t image_neck_out_channels = 0;
    std::vector<float> image_lss_xbound;
    std::vector<float> image_lss_ybound;
    std::vector<float> image_lss_zbound;
    std::vector<float> image_lss_dbound;
    int32_t image_lss_in_channels = 0;
    int32_t image_lss_out_channels = 0;
    int32_t image_lss_downsample = 0;
    int32_t image_fuser_in_channels = 0;
    int32_t image_fuser_out_channels = 0;

    // CaDDN monocular image/frustum branch.
    int32_t caddn_image_height = 0;
    int32_t caddn_image_width = 0;
    int32_t caddn_depth_bins = 0;
    float caddn_depth_min = 0.0f;
    float caddn_depth_max = 0.0f;
    int32_t caddn_depth_downsample = 0;
    std::string caddn_depth_mode;

    // --- post-processing ---
    float score_thresh = 0.1f;
    float nms_thresh = 0.01f;
    int nms_pre_maxsize = 4096;
    int max_boxes = 100;

    // --- numerical ---
    float bn_eps = 1e-3f;  // OpenPCDet BN default

    // PointNet2 and two-stage RoI metadata.
    int32_t sample_points = 0;
    std::vector<int32_t> pointnet_sa_npoints;
    std::vector<int32_t> pointnet_sa_radius_counts;
    std::vector<float> pointnet_sa_radii;
    std::vector<int32_t> pointnet_sa_nsamples;
    std::vector<int32_t> pointnet_sa_mlp_depths;
    std::vector<int32_t> pointnet_sa_mlp_channels;
    std::vector<int32_t> pointnet_fp_depths;
    std::vector<int32_t> pointnet_fp_channels;
    int32_t roi_point_sample_count = 0;
    std::vector<float> roi_point_extra_width;
    float roi_depth_normalizer = 1.0f;
    std::vector<int32_t> roi_sa_npoints;
    std::vector<float> roi_sa_radii;
    std::vector<int32_t> roi_sa_nsamples;
    std::vector<int32_t> roi_sa_mlp_depths;
    std::vector<int32_t> roi_sa_mlp_channels;
    std::vector<float> point_mean_sizes;
    float roi_proposal_nms_thresh = 0.7f;
    int32_t roi_proposal_pre_max = 1024;
    int32_t roi_proposal_post_max = 100;
    int32_t roi_grid_size = 0;
    std::string roi_kind;
    int32_t roi_grid_in_channels = 0;
    int32_t roi_grid_downsample_ratio = 0;
    std::vector<int32_t> roi_shared_fc_channels;
    std::vector<int32_t> roi_iou_fc_channels;
    std::vector<int32_t> roi_cls_fc_channels;
    std::vector<int32_t> roi_reg_fc_channels;
    int32_t pfe_num_keypoints = 0;
    int32_t pfe_num_output_features = 0;
    std::vector<std::string> pfe_sa_sources;
    std::vector<int32_t> pfe_sa_radius_counts;
    std::vector<float> pfe_sa_radii;
    std::vector<int32_t> pfe_sa_nsamples;
    std::vector<int32_t> pfe_sa_mlp_depths;
    std::vector<int32_t> pfe_sa_mlp_channels;
    std::vector<float> roi_grid_radii;
    std::vector<int32_t> roi_grid_nsamples;
    std::vector<int32_t> roi_grid_mlp_depths;
    std::vector<int32_t> roi_grid_mlp_channels;
    std::vector<float> roi_voxel_radii;
    std::vector<int32_t> roi_voxel_nsamples;
    std::vector<int32_t> roi_voxel_query_ranges;
    std::vector<int32_t> roi_voxel_source_counts;
    bool parta2_anchor_proposals = false;
    int32_t roi_aware_pool_size = 0;
    int32_t roi_aware_num_features = 0;
    int32_t roi_aware_max_points = 0;
    bool roi_aware_disable_part = false;
    float roi_aware_seg_threshold = 0.0f;
    std::vector<int32_t> point_head_cls_fc;
    std::vector<int32_t> point_head_part_fc;
    std::vector<int32_t> point_head_reg_fc;

    // --- focal / multimodal (Focal VoxelRCNN) ---
    bool focal_multimodal = false;
    int32_t focal_kernel_size = 3;
    int32_t focal_image_channel = 16;
    bool focal_topk = true;
    float focal_threshold = 0.5f;
    bool focal_mask_multi = false;
    bool focal_skip_mask_kernel = false;
    // Option C (DETERMINISTIC_FOCAL_DESIGN.md): run the whole focal chain
    // (conv_imp + main conv of every FocalSparseConv) through the shared
    // deterministic sparse-conv core instead of ggml mul_mat, so CPU/CUDA/
    // Vulkan agree bit-for-bit. Focal models default to true (set by the
    // loader); a GGUF "focal.deterministic" key can override. Other models
    // never read this field.
    bool focal_deterministic = false;

    // --- FastPillars backbone ---
    int32_t fastpillar_base_channels = 64;
    std::vector<int32_t> fastpillar_stage_blocks;  // e.g. [6, 6, 3, 1]

    // --- derived helpers ---
    int32_t num_anchors_per_location() const {
        // sum over sets of (|rot| * |size| * |height|)
        int32_t n = 0;
        for (size_t s = 0; s < anchor_sizes.size(); s++) {
            n += (int32_t)(anchor_rotations[s].size() * anchor_sizes[s].size() *
                           anchor_bottom_heights[s].size());
        }
        return n;
    }
};

struct model_file {
    gguf_context *guf = nullptr;
    ggml_context *ctx = nullptr;  // tensor metadata; tensor data iff with_data
    hparams hp{};
    std::string error;

    model_file() = default;
    model_file(const model_file &) = delete;
    model_file &operator=(const model_file &) = delete;
    ~model_file() { close(); }

    bool open(const std::string &path, bool with_data);
    void close();

    ggml_tensor *tensor(const char *name) const;  // nullptr if absent
    ggml_tensor *require(const char *name);       // sets error if absent
};

}  // namespace openpcdet
