#include "bev_backbone.h"

#include <ggml.h>

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <numeric>

namespace openpcdet {

namespace {

#ifdef AICORE_OPENPCDET_CUDA_EXT
// CUDA-ext fused BN-fold+ReLU flag (upstream bev_backbone.cpp g_bn_fuse_relu).
// Set by model::realize_weights once the backend is known: conv_bn_relu emits
// the single fused ggml_bn_fold_relu node on CUDA instead of the mul + add +
// relu triple (the CUDA kernel is ported as ggml patch 0024). Packed q8_0
// direct convolution stays on the im2col + GEMM path in this port.
bool g_bn_fuse_relu = false;
#endif

// Upper bound on the patch count (OW*OH*N) routed through ggml_conv_2d_direct.
// The CPU direct conv batches patches through a fixed-size work buffer
// (GGML_IM2COL_WORK_SIZE, 16MB); beyond ~5240 patches per batch it currently
// scatters results across output cells (value multiset preserved, positions
// scrambled). The CPU backend resets this to a conservative threshold in
// backend init; CUDA and other backends keep the fused direct path at any
// size (set_conv_direct_max_patches, called once from engine init).
int64_t g_conv_direct_max_patches = INT64_MAX;

int64_t conv_direct_max_patches() { return g_conv_direct_max_patches; }

// Apply per-channel BN-folded affine: y = x * scale + bias, broadcasting over
// the channel dim (ne[2]). x is [W, H, C, N].
ggml_tensor * apply_bn_fold(ggml_context * ctx, ggml_tensor * x,
                            ggml_tensor * scale, ggml_tensor * bias) {
    const int64_t C = x->ne[2];
    ggml_tensor * s = ggml_reshape_4d(ctx, scale, 1, 1, C, 1);
    ggml_tensor * b = ggml_reshape_4d(ctx, bias,  1, 1, C, 1);
    return ggml_add(ctx, ggml_mul(ctx, x, s), b);
}

#ifdef AICORE_OPENPCDET_CUDA_EXT
// conv -> BN-fold + ReLU (single fused op on CUDA-ext; mul + add + relu
// triple on every other backend).
ggml_tensor * conv_bn_relu_impl(ggml_context * ctx, ggml_tensor * conv,
                                ggml_tensor * scale, ggml_tensor * bias) {
    const bool fuse = g_bn_fuse_relu && conv->ne[3] == 1;
    if (fuse) {
        const int64_t C = conv->ne[2];
        ggml_tensor * s = ggml_reshape_4d(ctx, scale, 1, 1, C, 1);
        ggml_tensor * b = ggml_reshape_4d(ctx, bias,  1, 1, C, 1);
        return ggml_bn_fold_relu(ctx, conv, s, b);
    }
    return ggml_relu(ctx, apply_bn_fold(ctx, conv, scale, bias));
}
#else
ggml_tensor * conv_bn_relu_impl(ggml_context * ctx, ggml_tensor * conv,
                                ggml_tensor * scale, ggml_tensor * bias) {
    return ggml_relu(ctx, apply_bn_fold(ctx, conv, scale, bias));
}
#endif

ggml_tensor * conv_2d_any(ggml_context * ctx, ggml_tensor * weight,
                          ggml_tensor * x, int stride, int pad, int dilation = 1) {
    // F32 4-D kernels go through ggml_conv_2d_direct (GGML_OP_CONV_2D): on CUDA
    // this dispatches to a fused f32 direct conv kernel, several times faster
    // than im2col+GEMM and without the f16 im2col precision loss of
    // ggml_conv_2d. Only the (stride 1, dilation 1) case has a fast fused
    // kernel; strided/dilated F32 kernels keep the explicit im2col path (uses
    // cuBLAS GEMM, correct and fast for the few downsample convs).
    const int kw = weight->ne[0];
    const int kh = weight->ne[1];
    // Fused packed kernels (CenterHead branch fusion): ggml_concat always
    // emits 4-D, so concatenated q8_0 packed weights arrive as
    // [IC*K*K, OC, 1, N] with ne[2]==1 marking the packed layout (true 4-D
    // kernels have ne[2]==IC).
    const bool packed4 = ggml_n_dims(weight) == 4 && weight->ne[2] == 1;
    // Packed quantized kernels: 2-D [IC*K*K, OC] (1-D [IC*K*K] when ggml
    // trimmed a trailing OC==1 dim), or fused 4-D [IC*K*K, OC, 1, N].
    const bool packed = ggml_n_dims(weight) <= 2 || packed4;
    const bool fast_compatible = (stride == 1 || stride == 2) &&
                                 (dilation == 1 || dilation == 2) &&
                                 kw == kh && (kw == 1 || kw == 2 || kw == 3);
    // f32/f16/q8_0 4-D kernels (and single-batch q8/f16) go through
    // ggml_conv_2d_direct (GGML_OP_CONV_2D): on CUDA this dispatches to the
    // fused direct conv kernel which dequantizes f16/q8 weights in-kernel,
    // several times faster than im2col+GEMM and without ggml_conv_2d's f16
    // im2col precision loss. Only single-batch q8/f16 are routed here (the
    // fused kernel requires B==1); multi-batch f16/q8 keep the im2col path.
    const bool qtype = weight->type == GGML_TYPE_F32 ||
                       weight->type == GGML_TYPE_F16 ||
                       weight->type == GGML_TYPE_Q8_0;
    const bool single_batch = (ggml_n_dims(x) < 4) || (x->ne[3] == 1);
    // Output patch count for this conv (OW*OH*N): the direct kernel batches
    // patches through the work buffer, so large outputs fall back to the
    // im2col+GEMM path (see g_conv_direct_max_patches above).
    const int64_t ow = (x->ne[0] + 2 * pad - (int64_t) kw) / stride + 1;
    const int64_t oh = (x->ne[1] + 2 * pad - (int64_t) kh) / stride + 1;
    const int64_t patch_total = ow * oh * x->ne[3];
    const bool direct_allowed = patch_total <= conv_direct_max_patches();
    if (qtype && fast_compatible && single_batch && direct_allowed &&
        ggml_n_dims(weight) >= 3 && !packed4) {
        ggml_tensor * kern = weight;
        if (ggml_n_dims(weight) == 3)  // OC collapsed to 1 -> [KW,KH,IC,1]
            kern = ggml_reshape_4d(ctx, weight, kw, kh, weight->ne[2], 1);
        ggml_tensor * out =
                ggml_conv_2d_direct(ctx, kern, x, stride, stride, pad, pad,
                                    dilation, dilation);
        // CUDA conv marker: opts this F32 conv into the WMMA TF32 igemm route
        // (ggml-cuda conv2d_igemm node-name gate). The upstream engine runs
        // tensor-core conv arithmetic on CUDA as well, so detector parity is
        // judged against the upstream CUDA baselines; CPU/Vulkan and every
        // other task keep their existing arithmetic.
        ggml_set_name(out, "opcpd_conv");
        return out;
    }
    if (ggml_n_dims(weight) == 3 && weight->type != GGML_TYPE_F32) {
        return ggml_conv_2d(ctx, weight, x, stride, stride, pad, pad, dilation, dilation);
    }

    GGML_ASSERT(ggml_n_dims(weight) == 1 || ggml_n_dims(weight) == 2 ||
                ggml_n_dims(weight) == 3 ||
                ggml_n_dims(weight) == 4);
    const int64_t IC = x->ne[2];
    const int64_t K = packed ? weight->ne[0] : weight->ne[0] * weight->ne[1] * weight->ne[2];
    const int64_t OC = packed
        ? (ggml_n_dims(weight) == 2 ? weight->ne[1] : weight->ne[1] * weight->ne[3])
        : weight->ne[3];
    GGML_ASSERT(IC > 0 && K % IC == 0);
    const int64_t kernel_area = K / IC;
    const int64_t kernel = (int64_t) std::llround(std::sqrt((double) kernel_area));
    GGML_ASSERT(kernel * kernel == kernel_area);

    ggml_tensor * shape_kernel = packed
        ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kernel, kernel, IC, OC)
        : weight;
    ggml_tensor * im2col = ggml_im2col(ctx, shape_kernel, x,
                                       stride, stride, pad, pad, dilation, dilation,
                                       true, GGML_TYPE_F32);
    ggml_tensor * cols = ggml_reshape_2d(ctx, im2col, K,
                                         im2col->ne[1] * im2col->ne[2] * im2col->ne[3]);
    // Normalize the kernel to 2-D [K, OC] (no-op for packed 2-D; restores the
    // trimmed trailing dim for 1-D/3-D kernels; for fused 4-D packed the
    // ne[0]==K row layout is unchanged, only the trailing dims flatten) before
    // the GEMM.
    ggml_tensor * matrix = ggml_reshape_2d(ctx, weight, K, OC);
    ggml_tensor * mm = ggml_mul_mat(ctx, matrix, cols); // [OC, OW*OH*N]
    ggml_tensor * cwhn = ggml_reshape_4d(ctx, mm, OC, im2col->ne[1],
                                         im2col->ne[2], im2col->ne[3]);
    return ggml_cont(ctx, ggml_permute(ctx, cwhn, 2, 0, 1, 3));
}

// A conv2d + BN + ReLU block. conv_w is [KW, KH, IC, OC]. input is [W, H, IC, N].
// stride/pad are the conv params. Returns [W', H', OC, N].
// Uses ggml_conv_2d (handles im2col + matmul internally on the backend).
ggml_tensor * conv_bn_relu(ggml_context * ctx,
                           const conv_bn_block_weights & w,
                           ggml_tensor * x,
                           int stride, int pad) {
    // ggml_conv_2d(ctx, kernel, input, s0, s1, p0, p1, d0, d1)
    //   kernel: [KW, KH, IC, OC]; input: [W, H, IC, N]; out: [W', H', OC, N]
    ggml_tensor * conv = conv_2d_any(ctx, w.conv_w, x, stride, pad);
    return conv_bn_relu_impl(ctx, conv, w.bn_scale, w.bn_bias);
}

// A ConvTranspose2d + BN + ReLU deblock. conv_w is [KW, KH, IC, OC].
// input is [W, H, IC, N]. stride is the transposed conv stride.
// Returns [W*stride, H*stride, OC, N].
// ggml_conv_transpose_2d_p0 has no padding arg (always 0) — matches the
// OpenPCDet deblocks which use kernel_size == stride and padding == 0.
ggml_tensor * conv_transpose_bn_relu(ggml_context * ctx,
                                     const conv_bn_block_weights & w,
                                     ggml_tensor * x,
                                     int stride) {
    ggml_tensor * out = ggml_conv_transpose_2d_p0(ctx, w.conv_w, x, stride);
    // CUDA deconv marker: opts this F32 deblock conv into the byte-faithful
    // upstream OpenPCDet-GGML phase-decomposed GEMM path (ggml patch 0026).
    // The upstream engine runs the same TF32-class arithmetic on CUDA, so
    // detector parity is judged against the upstream CUDA baselines; CPU,
    // Vulkan and every other task keep their existing arithmetic.
    ggml_set_name(out, "opcpd_deconv");
    return conv_bn_relu_impl(ctx, out, w.bn_scale, w.bn_bias);
}

} // namespace

ggml_tensor * build_conv_bn_relu(ggml_context * ctx,
                                 const conv_bn_block_weights & weights,
                                 ggml_tensor * input, int stride, int padding) {
    return conv_bn_relu(ctx, weights, input, stride, padding);
}

void set_conv_direct_max_patches(int64_t max_patches) {
    g_conv_direct_max_patches = max_patches;
}

#ifdef AICORE_OPENPCDET_CUDA_EXT
void set_bn_fuse_relu(bool enable) {
    g_bn_fuse_relu = enable;
}
#endif

ggml_tensor * build_conv_bn_relu_dilated(ggml_context * ctx,
                                         const conv_bn_block_weights & weights,
                                         ggml_tensor * input, int stride,
                                         int padding, int dilation) {
    ggml_tensor * output = conv_2d_any(ctx, weights.conv_w, input, stride, padding, dilation);
    return conv_bn_relu_impl(ctx, output, weights.bn_scale, weights.bn_bias);
}

ggml_tensor * build_conv_bias(ggml_context * ctx, ggml_tensor * weight,
                              ggml_tensor * bias, ggml_tensor * input,
                              int stride, int padding) {
    ggml_tensor * out = conv_2d_any(ctx, weight, input, stride, padding);
    if (bias) {
        ggml_tensor * reshaped = ggml_reshape_4d(ctx, bias, 1, 1, out->ne[2], 1);
        out = ggml_add(ctx, out, reshaped);
    }
    return out;
}

ggml_tensor * build_conv_bias_dilated(ggml_context * ctx, ggml_tensor * weight,
                                      ggml_tensor * bias, ggml_tensor * input,
                                      int stride, int padding, int dilation) {
    ggml_tensor * out = conv_2d_any(ctx, weight, input, stride, padding, dilation);
    if (bias) {
        ggml_tensor * reshaped = ggml_reshape_4d(ctx, bias, 1, 1, out->ne[2], 1);
        out = ggml_add(ctx, out, reshaped);
    }
    return out;
}

bool load_bev_backbone_weights(model_file & file, const hparams & hp,
                               bev_backbone_weights & w, std::string & err) {
    const int num_levels = (int) hp.bb_layer_nums.size();
    if (num_levels == 0) { err = "backbone.layer_nums empty"; return false; }
    if ((int) hp.bb_layer_strides.size() != num_levels ||
        (int) hp.bb_num_filters.size()    != num_levels) {
        err = "backbone layer_nums/strides/filters length mismatch";
        return false;
    }

    auto check_conv = [&](ggml_tensor * tensor, int in_channels, int out_channels,
                          int kernel, const std::string & name) {
        const bool ok = ggml_n_dims(tensor) == 2
            ? tensor->ne[0] == (int64_t) in_channels * kernel * kernel &&
              tensor->ne[1] == out_channels
            : tensor->ne[0] == kernel && tensor->ne[1] == kernel &&
              tensor->ne[2] == in_channels && tensor->ne[3] == out_channels;
        if (!ok && err.empty()) err = "convolution weight shape mismatch: " + name;
        return ok;
    };

    w.levels.resize(num_levels);
    int in_c = hp.num_bev_features;
    for (int lvl = 0; lvl < num_levels; lvl++) {
        const int layer_num  = hp.bb_layer_nums[lvl];
        const int num_filter = hp.bb_num_filters[lvl];
        w.levels[lvl].resize(layer_num + 1);

        const std::string stem = "backbone.blocks." + std::to_string(lvl) + ".0.";
        w.levels[lvl][0].conv_w   = file.require((stem + "conv.weight").c_str());
        w.levels[lvl][0].bn_scale = file.require((stem + "bn_fold.scale").c_str());
        w.levels[lvl][0].bn_bias  = file.require((stem + "bn_fold.bias").c_str());
        if (!file.error.empty()) { err = file.error; return false; }
        if (!check_conv(w.levels[lvl][0].conv_w, in_c, num_filter, 3, stem + "conv.weight")) {
            return false;
        }
        in_c = num_filter;

        for (int k = 0; k < layer_num; k++) {
            const std::string blk = "backbone.blocks." + std::to_string(lvl) + "." +
                                     std::to_string(k + 1) + ".";
            w.levels[lvl][k + 1].conv_w   = file.require((blk + "conv.weight").c_str());
            w.levels[lvl][k + 1].bn_scale = file.require((blk + "bn_fold.scale").c_str());
            w.levels[lvl][k + 1].bn_bias  = file.require((blk + "bn_fold.bias").c_str());
            if (!file.error.empty()) { err = file.error; return false; }
            if (!check_conv(w.levels[lvl][k + 1].conv_w, num_filter, num_filter, 3,
                            blk + "conv.weight")) return false;
        }
    }

    const int num_deblocks = (int) hp.bb_upsample_strides.size();
    if (num_deblocks > 0 &&
        (int) hp.bb_num_upsample_filters.size() != num_deblocks) {
        err = "backbone upsample_strides/num_upsample_filters length mismatch";
        return false;
    }
    w.deblocks.resize(num_deblocks);
    for (int d = 0; d < num_deblocks; d++) {
        const std::string p = "backbone.deblocks." + std::to_string(d) + ".";
        w.deblocks[d].conv_w   = file.require((p + "conv.weight").c_str());
        w.deblocks[d].bn_scale = file.require((p + "bn_fold.scale").c_str());
        w.deblocks[d].bn_bias  = file.require((p + "bn_fold.bias").c_str());
        if (!file.error.empty()) { err = file.error; return false; }
        if (d < num_levels) {
            const float upsample = hp.bb_upsample_strides[d];
            const int kernel = upsample >= 1.0f
                ? std::max(1, (int) std::llround(upsample))
                : std::max(1, (int) std::llround(1.0f / upsample));
            ggml_tensor * tensor = w.deblocks[d].conv_w;
            const bool transpose = hp.bb_deblock_transpose.size() == w.deblocks.size()
                ? hp.bb_deblock_transpose[d] != 0 : upsample >= 1.0f;
            const bool ok = ggml_n_dims(tensor) == 4 && tensor->ne[0] == kernel &&
                tensor->ne[1] == kernel &&
                tensor->ne[2] == (transpose ? hp.bb_num_upsample_filters[d] : hp.bb_num_filters[d]) &&
                tensor->ne[3] == (transpose ? hp.bb_num_filters[d] : hp.bb_num_upsample_filters[d]);
            if (!ok) { err = "deblock weight shape mismatch: " + p + "conv.weight"; return false; }
        }
    }

    w.num_bev_features = num_deblocks > 0
        ? (int) std::accumulate(hp.bb_num_upsample_filters.begin(),
                                hp.bb_num_upsample_filters.end(), 0)
        : hp.bb_num_filters.back();
    return true;
}

ggml_tensor * build_bev_backbone_graph(ggml_context * ctx,
                                       const bev_backbone_weights & w,
                                       const hparams & hp,
                                       ggml_tensor * spatial,
                                       std::vector<ggml_tensor *> * taps) {
    ggml_tensor * x = spatial;
    std::vector<ggml_tensor *> ups;
    ups.reserve(w.levels.size());

    for (size_t lvl = 0; lvl < w.levels.size(); lvl++) {
        const int stride = hp.bb_layer_strides[lvl];
        // Stem (with the ZeroPad2d folded into pad=1)
        x = conv_bn_relu(ctx, w.levels[lvl][0], x, stride, /*pad=*/1);
        if (taps) {
            ggml_set_name(x, ("backbone.level_" + std::to_string(lvl) + ".block_0").c_str());
            ggml_set_output(x);
            taps->push_back(x);
        }
        // Follow-up blocks (k=3, pad=1, stride=1)
        for (size_t k = 1; k < w.levels[lvl].size(); k++) {
            x = conv_bn_relu(ctx, w.levels[lvl][k], x, /*stride=*/1, /*pad=*/1);
            if (taps) {
                ggml_set_name(x, ("backbone.level_" + std::to_string(lvl) +
                                  ".block_" + std::to_string(k)).c_str());
                ggml_set_output(x);
                taps->push_back(x);
            }
        }
        // Deblock (upsample)
        if (lvl < w.deblocks.size()) {
            const float upsample = hp.bb_upsample_strides[lvl];
            ggml_tensor * up = nullptr;
            const bool transpose = hp.bb_deblock_transpose.size() == w.deblocks.size()
                ? hp.bb_deblock_transpose[lvl] != 0 : upsample >= 1.0f;
            if (transpose) {
                up = conv_transpose_bn_relu(ctx, w.deblocks[lvl], x,
                    std::max(1, (int) std::llround(upsample)));
            } else {
                const int down_stride = std::max(1, (int) std::llround(1.0f / upsample));
                up = conv_bn_relu(ctx, w.deblocks[lvl], x, down_stride, /*pad=*/0);
            }
            if (taps) {
                ggml_set_name(up, ("backbone.level_" + std::to_string(lvl) + ".up").c_str());
                ggml_set_output(up);
                taps->push_back(up);
            }
            ups.push_back(up);
        } else {
            ups.push_back(x);
        }
    }

    // Concat all deblock outputs along the channel dim (ne[2]).
    if (ups.size() == 1) return ups[0];
    ggml_tensor * out = ups[0];
    for (size_t i = 1; i < ups.size(); i++) {
        out = ggml_concat(ctx, out, ups[i], 2);  // dim=2 = channel dim
    }

    // Optional extra deblock (when len(upsample_strides) > num_levels)
    if (w.deblocks.size() > w.levels.size()) {
        const float upsample = hp.bb_upsample_strides.back();
        const bool transpose = hp.bb_deblock_transpose.size() == w.deblocks.size()
            ? hp.bb_deblock_transpose.back() != 0 : upsample >= 1.0f;
        if (transpose) {
            out = conv_transpose_bn_relu(ctx, w.deblocks.back(), out,
                std::max(1, (int) std::llround(upsample)));
        } else {
            const int down_stride = std::max(1, (int) std::llround(1.0f / upsample));
            out = conv_bn_relu(ctx, w.deblocks.back(), out, down_stride, /*pad=*/0);
        }
    }
    return out;
}

} // namespace openpcdet
