#include "conv_fuser.h"

#include <ggml.h>

#include <cstdio>
#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

namespace openpcdet {

// ---- Weight loading ----

bool load_conv_fuser_weights(model_file & file, const hparams & hp,
                             conv_fuser_weights & w, std::string & error) {
    w.conv.conv_w   = file.require("fuser.conv.0.weight");
    w.conv.bn_scale = file.require("fuser.conv.1.eval_scale");
    w.conv.bn_bias  = file.require("fuser.conv.1.eval_bias");
    if (!file.error.empty()) { error = file.error; return false; }
    return true;
}

// ---- Forward ----

bool run_conv_fuser(engine_backend & backend, const hparams & hp,
                    const conv_fuser_weights & w,
                    const float * img_bev, int img_channels,
                    const float * lidar_bev, int lidar_channels,
                    int width, int height, int batch_size,
                    std::vector<float> & output,
                    const tap_sink & sink, std::string & error) {
    const int in_channels = img_channels + lidar_channels;
    const int out_channels = hp.image_fuser_out_channels;

    // Concatenate img_bev and lidar_bev along the channel dimension.
    // GGML layout: [W, H, C, B] → concat at dim 2 → [W, H, C_img+C_lidar, B]
    const size_t spatial = (size_t) width * height;
    std::vector<float> concat_data(spatial * in_channels * batch_size);

    // NOTE the two inputs use DIFFERENT memory layouts:
    //   img_bev  comes from the DepthLSS ggml graph → WHC planes
    //            [x + W*y + W*H*c] — per-channel contiguous, so the img half
    //            of the concat is a straight memcpy.
    //   lidar_bev comes from HeightCompression (voxel_backbone_runner) →
    //            HWC-interleaved [(y*W + x) * C + c] — a channel-scatter that
    //            is parallelized across rows (cache-hostile otherwise).
    const size_t img_floats = (size_t) img_channels * spatial;
    const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    const int rows_per = (int) ((height + hw_threads - 1) / hw_threads);
    std::vector<std::thread> pool;
    for (int b = 0; b < batch_size; b++) {
        std::memcpy(concat_data.data() + (size_t) b * in_channels * spatial,
                    img_bev + (size_t) b * img_channels * spatial,
                    img_floats * sizeof(float));
        for (unsigned t = 0; t < hw_threads; t++) {
            const int h0 = (int) (t * rows_per);
            const int h1 = std::min(height, (int) ((t + 1) * rows_per));
            if (h0 >= h1) break;
            pool.emplace_back([&, b, h0, h1] {
                for (int h = h0; h < h1; h++) {
                    for (int ww = 0; ww < width; ww++) {
                        size_t out_base = ((size_t) b * in_channels) * spatial
                                        + (size_t) h * width + ww;
                        size_t lid_base = ((size_t) b * lidar_channels) * spatial
                                        + ((size_t) h * width + ww) * lidar_channels;
                        for (int c = 0; c < lidar_channels; c++)
                            concat_data[out_base
                              + (size_t) (img_channels + c) * spatial] =
                                lidar_bev[lid_base + c];
                    }
                }
            });
        }
    }
    for (auto & worker : pool) worker.join();

    // Build graph: Conv2d(3×3, pad=1) → BN → ReLU
    const size_t graph_nodes = 128;
    ggml_init_params gp = {
        ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false),
        nullptr, /*no_alloc=*/true,
    };
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) { error = "ConvFuser graph ctx init failed"; return false; }

    ggml_tensor * input_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                width, height, in_channels, batch_size);
    ggml_set_name(input_t, "fuser_input");
    ggml_set_input(input_t);

    // Conv2d(3×3, pad=1) → BN → ReLU via the shared conv builder. This routes
    // through conv_2d_any, whose patch-count ceiling falls back to the exact
    // im2col+GEMM path on the CPU backend: the fuser conv produces
    // 180*180 = 32400 patches, far beyond what the CPU direct conv batches
    // correctly through its fixed 16MB work buffer (~5240 patches).
    // The ggml conv weight layout: [KW, KH, IC, OC] (torch [OC,IC,KH,KW]).
    ggml_tensor * out_t = build_conv_bn_relu(ctx, w.conv, input_t, 1, 1);
    ggml_set_name(out_t, "fuser_out");
    ggml_set_output(out_t);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_nodes, false);
    ggml_build_forward_expand(gf, out_t);

    if (!ggml_gallocr_alloc_graph(backend.galloc, gf)) {
        error = "ConvFuser graph alloc failed";
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(input_t, concat_data.data(), 0,
                            concat_data.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend.be, gf) != GGML_STATUS_SUCCESS) {
        error = "ConvFuser graph compute failed";
        ggml_free(ctx);
        return false;
    }

    // Readback
    size_t n = (size_t) width * height * out_channels * batch_size;
    output.resize(n);
    ggml_backend_tensor_get(out_t, output.data(), 0, n * sizeof(float));

    ggml_free(ctx);
    return true;
}

} // namespace openpcdet
