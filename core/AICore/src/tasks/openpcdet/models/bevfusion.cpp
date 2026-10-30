#include "bevfusion.h"

#include "voxel_backbone_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace openpcdet {

namespace {

} // anon ns

bool bevfusion::load(const std::string & gguf_path, const std::string & device, int n_threads) {
    // Load LiDAR branch + TransFusion head via parent
    if (!transfusion::load(gguf_path, device, n_threads)) return false;
    if (file.hp.arch != arch_kind::bevfusion) {
        error = "bevfusion: architecture mismatch";
        return false;
    }

    // Load camera branch weights
    std::string load_error;
    if (!load_swin_weights(file, file.hp, swin_w, load_error) ||
        !load_lss_fpn_weights(file, file.hp, fpn_w, load_error) ||
        !load_depth_lss_weights(file, file.hp, lss_w, load_error) ||
        !load_conv_fuser_weights(file, file.hp, fuser_w, load_error)) {
        error = load_error;
        return false;
    }

    // Remap camera weights to device context if non-CPU
    if (device_ctx) {
        auto remap = [&](ggml_tensor *& t) {
            if (t) t = ggml_get_tensor(device_ctx, ggml_get_name(t));
        };
        auto remap_conv = [&](conv_bn_block_weights & cb) {
            remap(cb.conv_w); remap(cb.bn_scale); remap(cb.bn_bias);
        };
        // SwinTransformer
        remap(swin_w.patch_embed_w); remap(swin_w.patch_embed_b);
        remap(swin_w.patch_norm_w); remap(swin_w.patch_norm_b);
        for (auto & stage : swin_w.stages) {
            for (auto & blk : stage.blocks) {
                remap(blk.norm1_w); remap(blk.norm1_b);
                remap(blk.attn.qkv_w); remap(blk.attn.qkv_b);
                remap(blk.attn.proj_w); remap(blk.attn.proj_b);
                remap(blk.attn.rel_pos_bias);
                remap(blk.norm2_w); remap(blk.norm2_b);
                remap(blk.ffn.fc1_w); remap(blk.ffn.fc1_b);
                remap(blk.ffn.fc2_w); remap(blk.ffn.fc2_b);
            }
            if (stage.has_downsample) {
                remap(stage.downsample.norm_w);
                remap(stage.downsample.norm_b);
                remap(stage.downsample.reduction_w);
            }
        }
        for (auto & t : swin_w.out_norm_w) remap(t);
        for (auto & t : swin_w.out_norm_b) remap(t);

        // FPN
        for (auto & lvl : fpn_w.levels) {
            remap_conv(lvl.lateral);
            remap_conv(lvl.fpn_conv);
        }

        // DepthLSS
        remap_conv(lss_w.dtrans_0); remap_conv(lss_w.dtrans_1); remap_conv(lss_w.dtrans_2);
        remap_conv(lss_w.dnet_0); remap_conv(lss_w.dnet_1);
        remap(lss_w.dnet_final_w); remap(lss_w.dnet_final_b);
        if (lss_w.has_downsample) {
            remap_conv(lss_w.down_0); remap_conv(lss_w.down_1); remap_conv(lss_w.down_2);
        }

        // ConvFuser
        remap_conv(fuser_w.conv);
    }
    return true;
}

bool bevfusion::forward(const float * points, int32_t n_points, int32_t point_dim,
                         std::vector<detection> & out_dets, const tap_sink & sink) {
    out_dets.clear();
    error = "bevfusion requires forward_scene() with camera images";
    return false;
}

bool bevfusion::forward_scene(const scene_input & scene,
                               std::vector<detection> & out_dets,
                               const tap_sink & sink) {
    out_dets.clear();
    const hparams & hp = file.hp;

    if (!scene.points || scene.n_points <= 0) {
        error = "bevfusion: no point cloud input";
        return false;
    }
    if (!scene.images || scene.num_cameras <= 0) {
        error = "bevfusion: no camera image input";
        return false;
    }

    const bool profile = std::getenv("OPCPD_PROFILE") != nullptr;
    auto lap_start = std::chrono::steady_clock::now();
    auto lap = [&](const char * name) {
        if (!profile) return;
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[profile] %-24s %8.2f ms\n", name,
                     std::chrono::duration<double, std::milli>(now - lap_start).count());
        lap_start = now;
    };

    // ---- Step 1: LiDAR branch — sparse backbone + HeightCompression ----
    voxel_backbone_result voxel;
    if (!run_voxel_backbone(be, hp, weights.sparse, scene.points,
                            scene.n_points, scene.point_dim,
                            false, voxel, sink, error))
        return false;
    if (voxel.spatial_features.empty()) return true;
    lap("lidar_backbone");

    // lidar_bev: [W, H, C, 1] from HeightCompression
    int lidar_W = voxel.width;
    int lidar_H = voxel.height;
    int lidar_C = hp.num_bev_features;  // 256

    // Dump lidar BEV (pre-fusion) as [B*H*W, C] row-major HWC. The buffer is
    // already HWC-interleaved [(y*W + x) * C + c] (see voxel_backbone_runner),
    // so the tap is a direct copy.
    if (sink) {
        sink("bevfusion.lidar_bev", voxel.spatial_features.data(),
             (int64_t) lidar_H * lidar_W, lidar_C);
    }

    // ---- Step 2: Camera branch ----
    const int num_cameras = hp.image_num_cameras;  // 6
    const int img_H = hp.image_input_size[0];      // 256
    const int img_W = hp.image_input_size[1];      // 704

    // 2a. SwinTransformer
    swin_output swin_out;
    if (!run_swin_transformer(be, hp, swin_w,
                              scene.images, num_cameras,
                              img_H, img_W,
                              swin_out, sink, error))
        return false;
    lap("swin");

    // 2b. GeneralizedLSSFPN
    lss_fpn_output fpn_out;
    if (!run_lss_fpn(be, hp, fpn_w,
                     swin_out.features, swin_out.widths, swin_out.heights,
                     swin_out.channels, num_cameras,
                     fpn_out, sink, error))
        return false;
    lap("fpn");

    // 2c. DepthLSSTransform
    // Need to project LiDAR points for sparse depth input
    std::vector<float> img_bev;
    int bev_W = 0, bev_H = 0;
    if (!run_depth_lss(be, hp, lss_w,
                       fpn_out.features[0].data(),
                       fpn_out.widths[0], fpn_out.heights[0],
                       /*depth_input=*/nullptr,
                       img_W, img_H,
                       scene.camera_intrinsics,
                       scene.camera_to_lidar,
                       scene.image_augmentation,
                       scene.lidar_augmentation,
                       scene.lidar_to_image,
                       scene.points, scene.n_points, scene.point_dim,
                       /*batch_size=*/1,
                       img_bev, bev_W, bev_H,
                       sink, error))
        return false;
    lap("depth_lss");

    // ---- Step 3: Fusion ----
    // img_bev: [bev_W, bev_H, out_channel, 1]
    // lidar_bev: [lidar_W, lidar_H, lidar_C, 1]
    // Need to ensure spatial dimensions match
    if (bev_W != lidar_W || bev_H != lidar_H) {
        error = "bevfusion: BEV spatial size mismatch between image and LiDAR";
        return false;
    }

    // Dump image BEV (DepthLSS output) as [B*H*W, C] row-major HWC.
    if (sink) {
        int C = hp.image_lss_out_channels;
        int64_t rows = (int64_t) bev_H * bev_W;
        std::vector<float> tap((size_t) rows * C);
        for (int h = 0; h < bev_H; h++)
          for (int w = 0; w < bev_W; w++)
            for (int c = 0; c < C; c++) {
                size_t src = (size_t) w + (size_t) bev_W * h + (size_t) bev_W * bev_H * c;
                size_t dst = ((size_t) h * bev_W + w) * C + c;
                tap[dst] = img_bev[src];
            }
        sink("bevfusion.lss", tap.data(), rows, C);
    }

    // img_bev is WHC planes (ggml graph output); lidar_bev is HWC-interleaved
    // (HeightCompression); run_conv_fuser handles the two layouts explicitly.


    std::vector<float> fused;
    if (!run_conv_fuser(be, hp, fuser_w,
                        img_bev.data(), hp.image_lss_out_channels,
                        voxel.spatial_features.data(), lidar_C,
                        lidar_W, lidar_H, /*batch_size=*/1,
                        fused, sink, error))
        return false;
    lap("fuser");

    // Dump fused BEV as [B*H*W, C] row-major HWC.
    if (sink) {
        int C = hp.image_fuser_out_channels;
        int64_t rows = (int64_t) lidar_H * lidar_W;
        std::vector<float> tap((size_t) rows * C);
        for (int h = 0; h < lidar_H; h++)
          for (int w = 0; w < lidar_W; w++)
            for (int c = 0; c < C; c++) {
                size_t src = (size_t) w + (size_t) lidar_W * h + (size_t) lidar_W * lidar_H * c;
                size_t dst = ((size_t) h * lidar_W + w) * C + c;
                tap[dst] = fused[src];
            }
        sink("bevfusion.fused", tap.data(), rows, C);
    }

    // ---- Step 4: TransFusion head ----
    // fused is already GGML WHC planes [x + W*y + W*H*c] — exactly what the
    // head graph consumes. Pass it through with input_is_whc so neither the
    // old whc_to_hwc nor forward_spatial's hwc_to_whc host transpose runs
    // (their composition is the identity on this buffer).
    int fused_W = lidar_W, fused_H = lidar_H;
    lap("fuser_readback");

    return forward_spatial(fused, fused_W, fused_H, out_dets, sink,
                           /*input_is_whc=*/true);
}

} // namespace openpcdet
