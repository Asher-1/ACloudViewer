#include "gguf_loader.h"

#include <algorithm>

namespace openpcdet {

namespace {

// KV readers: set err once on first failure, no-op afterwards.
struct kv {
    const gguf_context * g;
    std::string &        err;

    int64_t find(const char * key, bool required) const {
        int64_t id = gguf_find_key(g, key);
        if (id < 0 && required && err.empty()) {
            err = std::string("missing GGUF key: ") + key;
        }
        return id;
    }
    void u32(const char * key, int32_t & out, bool required = true) const {
        int64_t id = find(key, required);
        if (id >= 0) out = (int32_t) gguf_get_val_u32(g, id);
    }
    void i32(const char * key, int32_t & out, bool required = true) const {
        int64_t id = find(key, required);
        if (id >= 0) out = (int32_t) gguf_get_val_i32(g, id);
    }
    void f32(const char * key, float & out, bool required = true) const {
        int64_t id = find(key, required);
        if (id >= 0) out = gguf_get_val_f32(g, id);
    }
    void boolean(const char * key, bool & out, bool required = true) const {
        int64_t id = find(key, required);
        if (id >= 0) out = gguf_get_val_bool(g, id);
    }
    std::string str(const char * key, bool required = true) const {
        int64_t id = find(key, required);
        return id >= 0 ? gguf_get_val_str(g, id) : "";
    }
    void str_arr(const char * key, std::vector<std::string> & out,
                 bool required = true) const {
        int64_t id = find(key, required);
        if (id < 0) return;
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(g, id) != GGUF_TYPE_STRING) {
            if (err.empty()) err = std::string("expected string GGUF array: ") + key;
            return;
        }
        const size_t count = static_cast<size_t>(gguf_get_arr_n(g, id));
        out.resize(count);
        for (size_t i = 0; i < count; ++i) out[i] = gguf_get_arr_str(g, id, i);
    }
    // Read a fixed-length array of u32 / f32 into a std::vector.
    void u32_arr(const char * key, std::vector<int32_t> & out, bool required = true) const {
        int64_t id = find(key, required);
        if (id < 0) return;
        const int64_t n = gguf_get_arr_n(g, id);
        out.resize((size_t) std::max<int64_t>(0, n));
        const uint32_t * data = (const uint32_t *) gguf_get_arr_data(g, id);
        for (int64_t i = 0; i < n; i++) {
            out[i] = (int32_t) data[i];
        }
    }
    void i32_arr(const char * key, std::vector<int32_t> & out, bool required = true) const {
        int64_t id = find(key, required);
        if (id < 0) return;
        const int64_t n = gguf_get_arr_n(g, id);
        out.resize((size_t) std::max<int64_t>(0, n));
        const int32_t * data = (const int32_t *) gguf_get_arr_data(g, id);
        std::copy(data, data + n, out.begin());
    }
    void f32_arr(const char * key, std::vector<float> & out, bool required = true) const {
        int64_t id = find(key, required);
        if (id < 0) return;
        const int64_t n = gguf_get_arr_n(g, id);
        out.resize((size_t) std::max<int64_t>(0, n));
        const float * data = (const float *) gguf_get_arr_data(g, id);
        for (int64_t i = 0; i < n; i++) {
            out[i] = data[i];
        }
    }
    void number_arr(const char * key, std::vector<float> & out, bool required = true) const {
        int64_t id = find(key, required);
        if (id < 0) return;
        const int64_t n = gguf_get_arr_n(g, id);
        out.resize((size_t) std::max<int64_t>(0, n));
        const gguf_type type = gguf_get_arr_type(g, id);
        if (type == GGUF_TYPE_FLOAT32) {
            const float * data = (const float *) gguf_get_arr_data(g, id);
            std::copy(data, data + n, out.begin());
        } else if (type == GGUF_TYPE_UINT32) {
            const uint32_t * data = (const uint32_t *) gguf_get_arr_data(g, id);
            for (int64_t i = 0; i < n; i++) out[i] = (float) data[i];
        } else if (type == GGUF_TYPE_INT32) {
            const int32_t * data = (const int32_t *) gguf_get_arr_data(g, id);
            for (int64_t i = 0; i < n; i++) out[i] = (float) data[i];
        } else if (err.empty()) {
            err = std::string("expected f32/u32 GGUF array: ") + key;
        }
    }
};

static constexpr const char * ARCH_PP = "openpcdet.pointpillar";
static constexpr const char * ARCH_SECOND = "openpcdet.second";
static constexpr const char * ARCH_VOXEL = "openpcdet.voxel";
static constexpr const char * ARCH_VOXELNEXT = "openpcdet.voxelnext";
static constexpr const char * ARCH_POINTRCNN = "openpcdet.pointrcnn";
static constexpr const char * ARCH_PVRCNN = "openpcdet.pvrcnn";
static constexpr const char * ARCH_VOXELRCNN = "openpcdet.voxelrcnn";
static constexpr const char * ARCH_TRANSFUSION = "openpcdet.transfusion";
static constexpr const char * ARCH_PARTA2 = "openpcdet.parta2";
static constexpr const char * ARCH_BEVFUSION = "openpcdet.bevfusion";
static constexpr const char * ARCH_CADDN = "openpcdet.caddn";
static constexpr const char * ARCH_FOCAL_FORMER3D = "openpcdet.focal_former3d";
static constexpr const char * ARCH_FASTPILLARS = "openpcdet.fastpillars";

bool parse_detector(const kv & r, hparams & hp, const char * A, arch_kind arch);

bool parse_caddn(const kv & r, hparams & hp) {
    if (!parse_detector(r, hp, ARCH_CADDN, arch_kind::caddn)) return false;
    const std::string prefix = std::string(ARCH_CADDN) + ".";
    r.u32((prefix + "image_height").c_str(), hp.caddn_image_height);
    r.u32((prefix + "image_width").c_str(), hp.caddn_image_width);
    r.u32((prefix + "depth_bins").c_str(), hp.caddn_depth_bins);
    r.f32((prefix + "depth_min").c_str(), hp.caddn_depth_min);
    r.f32((prefix + "depth_max").c_str(), hp.caddn_depth_max);
    r.u32((prefix + "depth_downsample_factor").c_str(), hp.caddn_depth_downsample);
    hp.caddn_depth_mode = r.str((prefix + "depth_mode").c_str());
    if (hp.caddn_image_height <= 0 || hp.caddn_image_width <= 0 ||
        hp.caddn_depth_bins <= 1 || !(hp.caddn_depth_max > hp.caddn_depth_min) ||
        hp.caddn_depth_downsample <= 0 || hp.caddn_depth_mode != "LID") {
        if (r.err.empty()) r.err = "CaDDN image/depth metadata mismatch";
    }
    return r.err.empty();
}

bool parse_image_branch(const kv & r, hparams & hp, const char * A) {
    const std::string prefix = std::string(A) + ".image.";
    r.u32((prefix + "num_cameras").c_str(), hp.image_num_cameras);
    r.u32_arr((prefix + "input_size").c_str(), hp.image_input_size);
    r.u32_arr((prefix + "feature_size").c_str(), hp.image_feature_size);
    r.u32((prefix + "swin.embed_dims").c_str(), hp.image_swin_embed_dims);
    r.u32_arr((prefix + "swin.depths").c_str(), hp.image_swin_depths);
    r.u32_arr((prefix + "swin.num_heads").c_str(), hp.image_swin_num_heads);
    r.u32((prefix + "swin.window_size").c_str(), hp.image_swin_window_size);
    r.u32((prefix + "swin.mlp_ratio").c_str(), hp.image_swin_mlp_ratio);
    r.u32_arr((prefix + "swin.out_indices").c_str(), hp.image_swin_out_indices);
    r.u32_arr((prefix + "neck.in_channels").c_str(), hp.image_neck_in_channels);
    r.u32((prefix + "neck.out_channels").c_str(), hp.image_neck_out_channels);
    r.f32_arr((prefix + "lss.xbound").c_str(), hp.image_lss_xbound);
    r.f32_arr((prefix + "lss.ybound").c_str(), hp.image_lss_ybound);
    r.f32_arr((prefix + "lss.zbound").c_str(), hp.image_lss_zbound);
    r.f32_arr((prefix + "lss.dbound").c_str(), hp.image_lss_dbound);
    r.u32((prefix + "lss.in_channels").c_str(), hp.image_lss_in_channels);
    r.u32((prefix + "lss.out_channels").c_str(), hp.image_lss_out_channels);
    r.u32((prefix + "lss.downsample").c_str(), hp.image_lss_downsample);
    r.u32((prefix + "fuser.in_channels").c_str(), hp.image_fuser_in_channels);
    r.u32((prefix + "fuser.out_channels").c_str(), hp.image_fuser_out_channels);
    if (hp.image_num_cameras <= 0 || hp.image_input_size.size() != 2 ||
        hp.image_feature_size.size() != 2 || hp.image_swin_depths.size() != 4 ||
        hp.image_swin_num_heads.size() != 4 || hp.image_lss_xbound.size() != 3 ||
        hp.image_lss_ybound.size() != 3 || hp.image_lss_zbound.size() != 3 ||
        hp.image_lss_dbound.size() != 3) {
        if (r.err.empty()) r.err = "BEVFusion image metadata mismatch";
    }
    return r.err.empty();
}

bool parse_two_stage(const kv & r, hparams & hp, const char * A, arch_kind arch,
                     bool voxel_base = false) {
    hp.arch = arch;
    const std::string prefix(A);
    if (!voxel_base) {
        r.f32_arr((prefix + ".point_cloud_range").c_str(), hp.point_cloud_range);
        r.u32((prefix + ".num_point_features").c_str(), hp.num_point_features);
        r.u32((prefix + ".num_class").c_str(), hp.num_class);
    }
    if (arch == arch_kind::parta2) {
        r.f32_arr((prefix + ".voxel_size").c_str(), hp.voxel_size);
        r.u32_arr((prefix + ".grid_size").c_str(), hp.grid_size);
        r.u32((prefix + ".max_points_per_voxel").c_str(), hp.max_points_per_voxel);
        r.u32((prefix + ".max_voxels").c_str(), hp.max_voxels);
    }
    r.u32((prefix + ".sample_points").c_str(), hp.sample_points, false);
    r.i32_arr((prefix + ".pointnet2.sa.npoints").c_str(), hp.pointnet_sa_npoints, false);
    r.u32_arr((prefix + ".pointnet2.sa.radius_counts").c_str(), hp.pointnet_sa_radius_counts, false);
    r.f32_arr((prefix + ".pointnet2.sa.radii").c_str(), hp.pointnet_sa_radii, false);
    r.u32_arr((prefix + ".pointnet2.sa.nsamples").c_str(), hp.pointnet_sa_nsamples, false);
    r.u32_arr((prefix + ".pointnet2.sa.mlp_depths").c_str(), hp.pointnet_sa_mlp_depths, false);
    r.u32_arr((prefix + ".pointnet2.sa.mlp_channels").c_str(), hp.pointnet_sa_mlp_channels, false);
    r.u32_arr((prefix + ".pointnet2.fp.depths").c_str(), hp.pointnet_fp_depths, false);
    r.u32_arr((prefix + ".pointnet2.fp.channels").c_str(), hp.pointnet_fp_channels, false);
    r.u32((prefix + ".roi.point_pool.sample_count").c_str(), hp.roi_point_sample_count, false);
    r.f32_arr((prefix + ".roi.point_pool.extra_width").c_str(), hp.roi_point_extra_width, false);
    r.f32((prefix + ".roi.point_pool.depth_normalizer").c_str(), hp.roi_depth_normalizer, false);
    r.i32_arr((prefix + ".roi.sa.npoints").c_str(), hp.roi_sa_npoints, false);
    r.f32_arr((prefix + ".roi.sa.radii").c_str(), hp.roi_sa_radii, false);
    r.u32_arr((prefix + ".roi.sa.nsamples").c_str(), hp.roi_sa_nsamples, false);
    r.u32_arr((prefix + ".roi.sa.mlp_depths").c_str(), hp.roi_sa_mlp_depths, false);
    r.u32_arr((prefix + ".roi.sa.mlp_channels").c_str(), hp.roi_sa_mlp_channels, false);
    r.f32_arr((prefix + ".point_head.mean_sizes").c_str(), hp.point_mean_sizes, false);
    r.f32((prefix + ".roi.proposal.nms_thresh").c_str(), hp.roi_proposal_nms_thresh, false);
    r.u32((prefix + ".roi.proposal.pre_max").c_str(), hp.roi_proposal_pre_max, false);
    r.u32((prefix + ".roi.proposal.post_max").c_str(), hp.roi_proposal_post_max, false);
    r.u32((prefix + ".roi.grid_size").c_str(), hp.roi_grid_size, false);
    hp.roi_kind = r.str((prefix + ".roi.kind").c_str(), false);
    r.u32((prefix + ".roi.grid.in_channels").c_str(), hp.roi_grid_in_channels, false);
    r.u32((prefix + ".roi.grid.downsample_ratio").c_str(),
          hp.roi_grid_downsample_ratio, false);
    r.u32_arr((prefix + ".roi.shared_fc_channels").c_str(),
              hp.roi_shared_fc_channels, false);
    r.u32_arr((prefix + ".roi.iou_fc_channels").c_str(),
              hp.roi_iou_fc_channels, false);
    r.u32_arr((prefix + ".roi.cls_fc_channels").c_str(),
              hp.roi_cls_fc_channels, false);
    r.u32_arr((prefix + ".roi.reg_fc_channels").c_str(),
              hp.roi_reg_fc_channels, false);
    r.u32((prefix + ".pfe.num_keypoints").c_str(), hp.pfe_num_keypoints, false);
    r.u32((prefix + ".pfe.num_output_features").c_str(), hp.pfe_num_output_features, false);
    r.str_arr((prefix + ".pfe.sa.sources").c_str(), hp.pfe_sa_sources, false);
    r.u32_arr((prefix + ".pfe.sa.radius_counts").c_str(), hp.pfe_sa_radius_counts, false);
    r.f32_arr((prefix + ".pfe.sa.radii").c_str(), hp.pfe_sa_radii, false);
    r.u32_arr((prefix + ".pfe.sa.nsamples").c_str(), hp.pfe_sa_nsamples, false);
    r.u32_arr((prefix + ".pfe.sa.mlp_depths").c_str(), hp.pfe_sa_mlp_depths, false);
    r.u32_arr((prefix + ".pfe.sa.mlp_channels").c_str(), hp.pfe_sa_mlp_channels, false);
    r.f32_arr((prefix + ".roi.grid.radii").c_str(), hp.roi_grid_radii, false);
    r.u32_arr((prefix + ".roi.grid.nsamples").c_str(), hp.roi_grid_nsamples, false);
    r.u32_arr((prefix + ".roi.grid.mlp_depths").c_str(), hp.roi_grid_mlp_depths, false);
    r.u32_arr((prefix + ".roi.grid.mlp_channels").c_str(), hp.roi_grid_mlp_channels, false);
    r.f32_arr((prefix + ".roi.voxel.radii").c_str(), hp.roi_voxel_radii, false);
    r.u32_arr((prefix + ".roi.voxel.nsamples").c_str(), hp.roi_voxel_nsamples, false);
    r.u32_arr((prefix + ".roi.voxel.query_ranges").c_str(), hp.roi_voxel_query_ranges, false);
    r.u32_arr((prefix + ".roi.voxel.source_counts").c_str(), hp.roi_voxel_source_counts, false);
    r.boolean((prefix + ".parta2.anchor_proposals").c_str(),
              hp.parta2_anchor_proposals, false);
    r.u32((prefix + ".roi.aware.pool_size").c_str(), hp.roi_aware_pool_size, false);
    r.u32((prefix + ".roi.aware.num_features").c_str(), hp.roi_aware_num_features, false);
    r.u32((prefix + ".roi.aware.max_points_per_voxel").c_str(),
          hp.roi_aware_max_points, false);
    r.boolean((prefix + ".roi.aware.disable_part").c_str(),
              hp.roi_aware_disable_part, false);
    r.f32((prefix + ".roi.aware.seg_mask_score_thresh").c_str(),
          hp.roi_aware_seg_threshold, false);
    r.u32_arr((prefix + ".point_head.cls_fc").c_str(), hp.point_head_cls_fc, false);
    r.u32_arr((prefix + ".point_head.part_fc").c_str(), hp.point_head_part_fc, false);
    r.u32_arr((prefix + ".point_head.reg_fc").c_str(), hp.point_head_reg_fc, false);
    r.boolean((prefix + ".focal.multimodal").c_str(), hp.focal_multimodal, false);
    r.u32((prefix + ".focal.kernel_size").c_str(), hp.focal_kernel_size, false);
    r.u32((prefix + ".focal.image_channel").c_str(), hp.focal_image_channel, false);
    r.boolean((prefix + ".focal.topk").c_str(), hp.focal_topk, false);
    r.f32((prefix + ".focal.threshold").c_str(), hp.focal_threshold, false);
    r.boolean((prefix + ".focal.mask_multi").c_str(), hp.focal_mask_multi, false);
    r.boolean((prefix + ".focal.skip_mask_kernel").c_str(),
              hp.focal_skip_mask_kernel, false);
    // 方案 C (DETERMINISTIC_FOCAL_DESIGN.md): focal 模型默认走确定性链，GGUF
    // 键可显式覆盖（键缺省 = 跟随 focal.multimodal，非 focal 模型保持 false）。
    hp.focal_deterministic = hp.focal_multimodal;
    r.boolean((prefix + ".focal.deterministic").c_str(), hp.focal_deterministic,
              false);
    r.f32((prefix + ".post.score_thresh").c_str(), hp.score_thresh);
    r.f32((prefix + ".post.nms_thresh").c_str(), hp.nms_thresh);
    r.u32((prefix + ".post.nms_pre_maxsize").c_str(), hp.nms_pre_maxsize);
    r.u32((prefix + ".post.max_boxes").c_str(), hp.max_boxes);
    if (hp.point_cloud_range.size() != 6 || hp.num_point_features <= 0 || hp.num_class <= 0)
        r.err = "two-stage detector metadata mismatch";
    if (arch == arch_kind::pointrcnn &&
        (hp.sample_points <= 0 || hp.pointnet_sa_npoints.empty() ||
         hp.pointnet_fp_depths.empty() || hp.roi_point_sample_count <= 0 ||
         hp.roi_sa_npoints.empty() || hp.point_mean_sizes.size() !=
         static_cast<size_t>(hp.num_class) * 3))
        r.err = "PointRCNN metadata mismatch";
    if (arch == arch_kind::pvrcnn &&
        (hp.roi_grid_size <= 0 || hp.pfe_num_keypoints <= 0 ||
         hp.pfe_sa_sources.empty() || hp.pfe_sa_sources.size() != hp.pfe_sa_radius_counts.size() ||
         hp.pfe_sa_radii.size() != hp.pfe_sa_nsamples.size() ||
         hp.roi_grid_radii.empty() || hp.roi_grid_radii.size() != hp.roi_grid_nsamples.size()))
        r.err = "PVRCNN PFE/RoI metadata mismatch";
    if (arch == arch_kind::voxelrcnn &&
        (hp.roi_grid_size <= 0 || hp.roi_voxel_radii.empty() ||
         hp.roi_voxel_radii.size() != hp.roi_voxel_nsamples.size() ||
         hp.roi_voxel_query_ranges.size() != hp.roi_voxel_radii.size() * 3 ||
         hp.roi_voxel_source_counts.empty()))
        r.err = "VoxelRCNN RoI metadata mismatch";
    if (arch == arch_kind::second && hp.roi_kind == "SECONDHead" &&
        (hp.roi_grid_size <= 1 || hp.roi_grid_in_channels <= 0 ||
         hp.roi_grid_downsample_ratio <= 0 || hp.roi_shared_fc_channels.empty() ||
         hp.roi_iou_fc_channels.empty()))
        r.err = "SECOND-IoU RoI metadata mismatch";
    if (arch == arch_kind::parta2 &&
        (hp.roi_kind != "PartA2FCHead" || hp.roi_aware_pool_size <= 0 ||
         hp.roi_aware_num_features <= 0 || hp.roi_aware_max_points <= 0 ||
         hp.voxel_size.size() != 3 || hp.grid_size.size() != 3))
        r.err = "PartA2 sparse U-Net/RoI metadata mismatch";
    return r.err.empty();
}

// Parse the nested anchor generator config. Stored in the GGUF as parallel
// arrays under "openpcdet.pointpillar.anchor.*":
//   anchor.num_sets          : u32
//   anchor.set_sizes_flat    : f32[sum(sizes*3)]   (sizes per set concatenated)
//   anchor.set_sizes_count   : u32[num_sets]       (how many sizes in each set)
//   anchor.set_rotations_flat: f32[sum(rots)]      (rotations per set concatenated)
//   anchor.set_rotations_count: u32[num_sets]
//   anchor.set_heights_flat  : f32[sum(heights)]
//   anchor.set_heights_count : u32[num_sets]
//   anchor.feature_map_strides: u32[num_sets]
void parse_anchor_cfg(const kv & r, hparams & hp, const std::string & arch_key) {
    int32_t num_sets = 0;
    r.u32((arch_key + ".anchor.num_sets").c_str(), num_sets, false);
    if (num_sets <= 0) return;

    std::vector<int32_t> sizes_count, rots_count, hts_count;
    std::vector<float>   sizes_flat, rots_flat, hts_flat;
    r.u32_arr((arch_key + ".anchor.set_sizes_count").c_str(),    sizes_count);
    r.u32_arr((arch_key + ".anchor.set_rotations_count").c_str(),rots_count);
    r.u32_arr((arch_key + ".anchor.set_heights_count").c_str(),  hts_count);
    r.f32_arr((arch_key + ".anchor.set_sizes_flat").c_str(),     sizes_flat);
    r.f32_arr((arch_key + ".anchor.set_rotations_flat").c_str(), rots_flat);
    r.f32_arr((arch_key + ".anchor.set_heights_flat").c_str(),   hts_flat);
    r.u32_arr((arch_key + ".anchor.feature_map_strides").c_str(),hp.feature_map_strides);
    r.u32_arr((arch_key + ".anchor.align_center").c_str(), hp.anchor_align_center);

    if ((int) sizes_count.size() != num_sets ||
        (int) rots_count.size() != num_sets ||
        (int) hts_count.size() != num_sets ||
        (int) hp.feature_map_strides.size() != num_sets ||
        (int) hp.anchor_align_center.size() != num_sets) {
        if (r.err.empty()) r.err = "anchor metadata length mismatch";
        return;
    }

    hp.anchor_sizes.resize(num_sets);
    hp.anchor_rotations.resize(num_sets);
    hp.anchor_bottom_heights.resize(num_sets);

    size_t si = 0, ri = 0, hi = 0;
    for (int s = 0; s < num_sets; s++) {
        const int ns = sizes_count[s], nr = rots_count[s], nh = hts_count[s];
        if (ns < 0 || nr < 0 || nh < 0 ||
            si + (size_t) ns * 3 > sizes_flat.size() ||
            ri + (size_t) nr > rots_flat.size() ||
            hi + (size_t) nh > hts_flat.size()) {
            if (r.err.empty()) r.err = "anchor metadata data/count mismatch";
            return;
        }
        hp.anchor_sizes[s].resize(ns);
        for (int i = 0; i < ns; i++) {
            hp.anchor_sizes[s][i] = {sizes_flat[si], sizes_flat[si+1], sizes_flat[si+2]};
            si += 3;
        }
        hp.anchor_rotations[s].assign(rots_flat.begin() + ri,
                                      rots_flat.begin() + ri + nr);
        ri += nr;
        hp.anchor_bottom_heights[s].assign(hts_flat.begin() + hi,
                                           hts_flat.begin() + hi + nh);
        hi += nh;
    }
}

bool parse_detector(const kv & r, hparams & hp, const char * A, arch_kind arch) {
    hp.arch = arch;

    r.f32_arr((std::string(A) + ".point_cloud_range").c_str(),  hp.point_cloud_range);
    r.f32_arr((std::string(A) + ".voxel_size").c_str(),         hp.voxel_size);
    r.u32_arr((std::string(A) + ".grid_size").c_str(),          hp.grid_size);
    r.u32   ((std::string(A) + ".max_points_per_voxel").c_str(),hp.max_points_per_voxel);
    r.u32   ((std::string(A) + ".max_voxels").c_str(),          hp.max_voxels);
    r.u32   ((std::string(A) + ".num_point_features").c_str(),  hp.num_point_features);
    r.boolean((std::string(A) + ".use_absolute_xyz").c_str(),  hp.use_absolute_xyz);
    r.boolean((std::string(A) + ".with_distance").c_str(),     hp.with_distance);
    r.u32_arr((std::string(A) + ".vfe.num_filters").c_str(),    hp.vfe_num_filters);
    r.boolean((std::string(A) + ".vfe.use_norm").c_str(),       hp.vfe_use_norm);
    r.boolean((std::string(A) + ".vfe.dynamic").c_str(),        hp.vfe_dynamic, false);
    r.boolean((std::string(A) + ".sparse.residual").c_str(),    hp.sparse_residual, false);
    r.u32_arr((std::string(A) + ".sparse.channels").c_str(),    hp.sparse_channels, false);
    r.u32_arr((std::string(A) + ".sparse.kernel_sizes").c_str(),hp.sparse_kernel_sizes, false);
    r.u32((std::string(A) + ".sparse.out_channels").c_str(),    hp.sparse_out_channels, false);
    r.u32   ((std::string(A) + ".num_bev_features").c_str(),    hp.num_bev_features);
    r.u32_arr((std::string(A) + ".backbone.layer_nums").c_str(),        hp.bb_layer_nums);
    r.u32_arr((std::string(A) + ".backbone.layer_strides").c_str(),     hp.bb_layer_strides);
    r.u32_arr((std::string(A) + ".backbone.num_filters").c_str(),       hp.bb_num_filters);
    r.number_arr((std::string(A) + ".backbone.upsample_strides").c_str(), hp.bb_upsample_strides);
    r.u32_arr((std::string(A) + ".backbone.num_upsample_filters").c_str(),hp.bb_num_upsample_filters);
    r.u32_arr((std::string(A) + ".backbone.deblock_transpose").c_str(), hp.bb_deblock_transpose, false);
    r.u32   ((std::string(A) + ".head.num_class").c_str(),      hp.num_class);
    int32_t head_kind = 0;
    r.u32   ((std::string(A) + ".head.kind").c_str(), head_kind, false);
    hp.head_kind = head_kind == 4 ? dense_head_kind::transfusion :
                   head_kind == 3 ? dense_head_kind::voxelnext :
                   head_kind == 2 ? dense_head_kind::center :
                   head_kind == 1 ? dense_head_kind::anchor_multi : dense_head_kind::anchor_single;
    r.u32   ((std::string(A) + ".head.num_dir_bins").c_str(),   hp.num_dir_bins);
    r.boolean((std::string(A) + ".head.use_direction_classifier").c_str(), hp.use_direction_classifier);
    r.u32   ((std::string(A) + ".head.code_size").c_str(),      hp.code_size);
    r.f32   ((std::string(A) + ".head.dir_offset").c_str(),     hp.dir_offset);
    r.f32   ((std::string(A) + ".head.dir_limit_offset").c_str(),hp.dir_limit_offset);
    r.boolean((std::string(A) + ".head.encode_angle_by_sincos").c_str(), hp.encode_angle_by_sincos);
    r.boolean((std::string(A) + ".post.multi_class_nms").c_str(), hp.multi_class_nms, false);
    if (hp.head_kind == dense_head_kind::transfusion) {
        const std::string prefix = std::string(A) + ".head.transfusion.";
        r.u32((prefix + "num_proposals").c_str(), hp.transfusion_num_proposals);
        r.u32((prefix + "hidden_channels").c_str(), hp.transfusion_hidden_channels);
        r.u32((prefix + "num_heads").c_str(), hp.transfusion_num_heads);
        r.u32((prefix + "ffn_channels").c_str(), hp.transfusion_ffn_channels);
        r.u32((prefix + "nms_kernel_size").c_str(), hp.transfusion_nms_kernel_size);
        r.u32((prefix + "feature_map_stride").c_str(), hp.transfusion_feature_map_stride);
        r.u32((prefix + "middle_channels").c_str(), hp.transfusion_middle_channels);
        r.boolean((prefix + "has_velocity").c_str(), hp.transfusion_has_velocity);
        r.f32_arr((prefix + "post_center_range").c_str(), hp.transfusion_post_center_range);
    } else if (hp.head_kind == dense_head_kind::anchor_multi) {
        r.u32((std::string(A) + ".head.multi.num_heads").c_str(), hp.multi_num_heads);
        r.u32((std::string(A) + ".head.multi.shared_channels").c_str(), hp.multi_shared_channels);
        r.boolean((std::string(A) + ".head.multi.separate_reg").c_str(), hp.multi_separate_reg);
        r.u32((std::string(A) + ".head.multi.middle_convs").c_str(), hp.multi_middle_convs);
        r.u32((std::string(A) + ".head.multi.middle_channels").c_str(), hp.multi_middle_channels);
        r.u32_arr((std::string(A) + ".head.multi.class_counts").c_str(), hp.multi_head_class_counts);
        r.u32_arr((std::string(A) + ".head.multi.anchor_counts").c_str(), hp.multi_head_anchor_counts);
        r.u32_arr((std::string(A) + ".head.multi.label_indices").c_str(), hp.multi_label_indices);
        r.u32_arr((std::string(A) + ".head.multi.anchor_set_indices").c_str(), hp.multi_anchor_set_indices);
        r.u32_arr((std::string(A) + ".head.multi.reg_channels").c_str(), hp.multi_reg_channels);
    } else if (hp.head_kind == dense_head_kind::center ||
               hp.head_kind == dense_head_kind::voxelnext) {
        r.u32((std::string(A) + ".head.center.num_heads").c_str(), hp.center_num_heads);
        r.u32((std::string(A) + ".head.center.shared_channels").c_str(), hp.center_shared_channels);
        r.u32((std::string(A) + ".head.center.middle_convs").c_str(), hp.center_middle_convs);
        r.u32((std::string(A) + ".head.center.feature_map_stride").c_str(), hp.center_feature_map_stride);
        r.u32((std::string(A) + ".head.center.max_objects").c_str(), hp.center_max_objects);
        r.boolean((std::string(A) + ".head.center.has_velocity").c_str(), hp.center_has_velocity);
        r.boolean((std::string(A) + ".head.center.has_iou").c_str(), hp.center_has_iou, false);
        r.u32((std::string(A) + ".head.center.kernel_size").c_str(),
              hp.center_head_kernel_size, false);
        r.boolean((std::string(A) + ".head.center.double_flip").c_str(),
                  hp.center_double_flip, false);
        r.u32_arr((std::string(A) + ".head.center.class_counts").c_str(), hp.center_class_counts);
        r.u32_arr((std::string(A) + ".head.center.label_indices").c_str(), hp.center_label_indices);
        r.f32_arr((std::string(A) + ".head.center.post_limit_range").c_str(), hp.center_post_limit_range);
        r.f32_arr((std::string(A) + ".head.center.rectifier").c_str(), hp.center_rectifier, false);
        r.number_arr((std::string(A) + ".head.center.nms_thresholds").c_str(),
                     hp.center_nms_thresholds, false);
        r.u32_arr((std::string(A) + ".head.center.nms_pre_maxsizes").c_str(),
                  hp.center_nms_pre_maxsizes, false);
        r.u32_arr((std::string(A) + ".head.center.nms_post_maxsizes").c_str(),
                  hp.center_nms_post_maxsizes, false);
    }
    r.f32   ((std::string(A) + ".post.score_thresh").c_str(),   hp.score_thresh);
    r.f32   ((std::string(A) + ".post.nms_thresh").c_str(),     hp.nms_thresh);
    r.u32   ((std::string(A) + ".post.nms_pre_maxsize").c_str(),hp.nms_pre_maxsize, false);
    r.u32   ((std::string(A) + ".post.max_boxes").c_str(),      hp.max_boxes);
    r.f32   ((std::string(A) + ".bn_eps").c_str(),              hp.bn_eps);

    if (hp.head_kind != dense_head_kind::transfusion) parse_anchor_cfg(r, hp, A);

    // sanity checks
    if (hp.point_cloud_range.size() != 6) r.err = "point_cloud_range must be 6";
    else if (hp.voxel_size.size() != 3)   r.err = "voxel_size must be 3";
    else if (hp.grid_size.size() != 3)    r.err = "grid_size must be 3";
    else if (hp.vfe_num_filters.empty())  r.err = "vfe.num_filters empty";
    else if (arch != arch_kind::voxelnext && hp.bb_layer_nums.empty())
        r.err = "backbone.layer_nums empty";
    else if (arch == arch_kind::voxelnext &&
             (hp.sparse_channels.size() != 5 || hp.sparse_kernel_sizes.size() != 4 ||
              hp.sparse_out_channels <= 0))
        r.err = "VoxelNeXt sparse backbone metadata mismatch";
    else if (hp.head_kind != dense_head_kind::center &&
             hp.head_kind != dense_head_kind::voxelnext &&
             hp.head_kind != dense_head_kind::transfusion && hp.anchor_sizes.empty())
        r.err = "anchor config empty";
    else if (hp.head_kind == dense_head_kind::transfusion &&
             (hp.transfusion_num_proposals <= 0 || hp.transfusion_hidden_channels <= 0 ||
              hp.transfusion_num_heads <= 0 || hp.transfusion_ffn_channels <= 0 ||
              hp.transfusion_feature_map_stride <= 0 ||
              hp.transfusion_post_center_range.size() != 6))
        r.err = "TransFusion metadata mismatch";
    else if (hp.head_kind == dense_head_kind::anchor_multi &&
             (hp.multi_num_heads <= 0 ||
              (int) hp.multi_head_class_counts.size() != hp.multi_num_heads ||
              (int) hp.multi_head_anchor_counts.size() != hp.multi_num_heads)) {
        r.err = "AnchorHeadMulti metadata mismatch";
    } else if ((hp.head_kind == dense_head_kind::center ||
                hp.head_kind == dense_head_kind::voxelnext) &&
               (hp.center_num_heads <= 0 ||
                static_cast<int>(hp.center_class_counts.size()) != hp.center_num_heads ||
                hp.center_label_indices.size() != static_cast<size_t>(hp.num_class) ||
                hp.center_post_limit_range.size() != 6)) {
        r.err = "CenterHead metadata mismatch";
    }
    return r.err.empty();
}

} // namespace

bool model_file::open(const std::string & path, bool with_data) {
    close();

    gguf_init_params params = { /*no_alloc =*/ !with_data, /*ctx =*/ &ctx };
    guf = gguf_init_from_file(path.c_str(), params);
    if (!guf) {
        error = "failed to read GGUF: " + path;
        return false;
    }

    kv r{guf, error};
    const std::string arch = r.str("general.architecture");
    if (error.empty()) {
        if (arch == ARCH_PP) {
            parse_detector(r, hp, ARCH_PP, arch_kind::pointpillar);
        } else if (arch == ARCH_SECOND) {
            parse_detector(r, hp, ARCH_SECOND, arch_kind::second);
            if (error.empty() && !r.str((std::string(ARCH_SECOND) + ".roi.kind").c_str(), false).empty())
                parse_two_stage(r, hp, ARCH_SECOND, arch_kind::second, true);
        } else if (arch == ARCH_VOXEL) {
            parse_detector(r, hp, ARCH_VOXEL, arch_kind::voxel_detector);
        } else if (arch == ARCH_VOXELNEXT) {
            parse_detector(r, hp, ARCH_VOXELNEXT, arch_kind::voxelnext);
        } else if (arch == ARCH_POINTRCNN) {
            parse_two_stage(r, hp, ARCH_POINTRCNN, arch_kind::pointrcnn);
        } else if (arch == ARCH_PVRCNN) {
            parse_detector(r, hp, ARCH_PVRCNN, arch_kind::pvrcnn);
            if (error.empty())
                parse_two_stage(r, hp, ARCH_PVRCNN, arch_kind::pvrcnn, true);
        } else if (arch == ARCH_VOXELRCNN) {
            parse_detector(r, hp, ARCH_VOXELRCNN, arch_kind::voxelrcnn);
            if (error.empty())
                parse_two_stage(r, hp, ARCH_VOXELRCNN, arch_kind::voxelrcnn, true);
        } else if (arch == ARCH_TRANSFUSION) {
            parse_detector(r, hp, ARCH_TRANSFUSION, arch_kind::transfusion);
        } else if (arch == ARCH_PARTA2) {
            const bool anchored = [&] {
                bool value = false;
                r.boolean((std::string(ARCH_PARTA2) + ".parta2.anchor_proposals").c_str(),
                          value, false);
                return value;
            }();
            if (anchored) {
                parse_detector(r, hp, ARCH_PARTA2, arch_kind::parta2);
                if (error.empty())
                    parse_two_stage(r, hp, ARCH_PARTA2, arch_kind::parta2, true);
            } else {
                parse_two_stage(r, hp, ARCH_PARTA2, arch_kind::parta2);
            }
        } else if (arch == ARCH_BEVFUSION) {
            parse_detector(r, hp, ARCH_BEVFUSION, arch_kind::bevfusion);
            if (error.empty()) parse_image_branch(r, hp, ARCH_BEVFUSION);
        } else if (arch == ARCH_CADDN) {
            parse_caddn(r, hp);
        } else if (arch == ARCH_FOCAL_FORMER3D) {
            parse_detector(r, hp, ARCH_FOCAL_FORMER3D, arch_kind::focal_former3d);
            if (error.empty()) {
                // Parse focal sparse-conv hyperparameters (same keys as VoxelRCNN-focal).
                const std::string prefix(ARCH_FOCAL_FORMER3D);
                r.boolean((prefix + ".focal.multimodal").c_str(), hp.focal_multimodal, false);
                r.u32((prefix + ".focal.kernel_size").c_str(), hp.focal_kernel_size, false);
                r.u32((prefix + ".focal.image_channel").c_str(), hp.focal_image_channel, false);
                r.boolean((prefix + ".focal.topk").c_str(), hp.focal_topk, false);
                r.f32((prefix + ".focal.threshold").c_str(), hp.focal_threshold, false);
                r.boolean((prefix + ".focal.mask_multi").c_str(), hp.focal_mask_multi, false);
                r.boolean((prefix + ".focal.skip_mask_kernel").c_str(),
                          hp.focal_skip_mask_kernel, false);
                // Same deterministic-focal default as parse_two_stage.
                hp.focal_deterministic = hp.focal_multimodal;
                r.boolean((prefix + ".focal.deterministic").c_str(),
                          hp.focal_deterministic, false);
            }
        } else if (arch == ARCH_FASTPILLARS) {
            parse_detector(r, hp, ARCH_FASTPILLARS, arch_kind::fastpillars);
            if (error.empty()) {
                const std::string prefix(ARCH_FASTPILLARS);
                r.u32((prefix + ".fastpillar.base_channels").c_str(),
                      hp.fastpillar_base_channels, false);
                r.i32_arr((prefix + ".fastpillar.stage_blocks").c_str(),
                          hp.fastpillar_stage_blocks);
            }
        } else {
            error = "unsupported architecture '" + arch + "'";
        }
    }

    if (!error.empty()) {
        const std::string saved_error = error;
        close();
        error = saved_error;
        return false;
    }
    return true;
}

void model_file::close() {
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
    if (guf) { gguf_free(guf); guf = nullptr; }
    hp = hparams{};
    error.clear();
}

ggml_tensor * model_file::tensor(const char * name) const {
    return ctx ? ggml_get_tensor(ctx, name) : nullptr;
}

ggml_tensor * model_file::require(const char * name) {
    ggml_tensor * t = tensor(name);
    if (!t && error.empty()) {
        error = std::string("missing tensor: ") + name;
    }
    return t;
}

} // namespace openpcdet
