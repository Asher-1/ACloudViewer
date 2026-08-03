#include "arcface_graph.hpp"
#include "antispoof_graph.hpp"
#include "backend.hpp"
#include "graph_ops.hpp"
#include "directconv.hpp"
#include "preprocess.hpp"
#include "model_loader.hpp"
#include "common.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace fd {

namespace {
// ArcFace IResNet stem tensor names (verbatim GGUF names from
// tensor_manifest_buffalo_l.md / the w600k_r50 ONNX graph): the first conv
// (rec.685 weight, rec.686 folded-BN bias) and the stem PReLU slope (rec.843,
// ggml ne [1,1,64]).
constexpr const char* kStemConvW    = "rec.685";
constexpr const char* kStemConvB    = "rec.686";
constexpr const char* kStemPReluA   = "rec.843";

// ONNX BatchNormalization epsilon for w600k_r50 (attribute on every bn1 node).
constexpr float kRecBnEps = 1e-5f;

// Verbatim per-block tensor names for the 24 IResNet50 IBasicBlocks, grouped by
// stage ([3,4,14,3]). Extracted node-by-node from w600k_r50.onnx (a stage's
// first block has stride 2 + a downsample conv; the rest stride 1, no shortcut).
// Every name was checked to exist in the buffalo_l GGUF.
const RecBlock kRecBlocks[] = {
  // layer1: 3 blocks
  { "rec.layer1.0.bn1", "rec.688", "rec.689", "rec.844", "rec.691", "rec.692", "rec.694", "rec.695", 2 },
  { "rec.layer1.1.bn1", "rec.697", "rec.698", "rec.845", "rec.700", "rec.701", nullptr, nullptr, 1 },
  { "rec.layer1.2.bn1", "rec.703", "rec.704", "rec.846", "rec.706", "rec.707", nullptr, nullptr, 1 },
  // layer2: 4 blocks
  { "rec.layer2.0.bn1", "rec.709", "rec.710", "rec.847", "rec.712", "rec.713", "rec.715", "rec.716", 2 },
  { "rec.layer2.1.bn1", "rec.718", "rec.719", "rec.848", "rec.721", "rec.722", nullptr, nullptr, 1 },
  { "rec.layer2.2.bn1", "rec.724", "rec.725", "rec.849", "rec.727", "rec.728", nullptr, nullptr, 1 },
  { "rec.layer2.3.bn1", "rec.730", "rec.731", "rec.850", "rec.733", "rec.734", nullptr, nullptr, 1 },
  // layer3: 14 blocks
  { "rec.layer3.0.bn1", "rec.736", "rec.737", "rec.851", "rec.739", "rec.740", "rec.742", "rec.743", 2 },
  { "rec.layer3.1.bn1", "rec.745", "rec.746", "rec.852", "rec.748", "rec.749", nullptr, nullptr, 1 },
  { "rec.layer3.2.bn1", "rec.751", "rec.752", "rec.853", "rec.754", "rec.755", nullptr, nullptr, 1 },
  { "rec.layer3.3.bn1", "rec.757", "rec.758", "rec.854", "rec.760", "rec.761", nullptr, nullptr, 1 },
  { "rec.layer3.4.bn1", "rec.763", "rec.764", "rec.855", "rec.766", "rec.767", nullptr, nullptr, 1 },
  { "rec.layer3.5.bn1", "rec.769", "rec.770", "rec.856", "rec.772", "rec.773", nullptr, nullptr, 1 },
  { "rec.layer3.6.bn1", "rec.775", "rec.776", "rec.857", "rec.778", "rec.779", nullptr, nullptr, 1 },
  { "rec.layer3.7.bn1", "rec.781", "rec.782", "rec.858", "rec.784", "rec.785", nullptr, nullptr, 1 },
  { "rec.layer3.8.bn1", "rec.787", "rec.788", "rec.859", "rec.790", "rec.791", nullptr, nullptr, 1 },
  { "rec.layer3.9.bn1", "rec.793", "rec.794", "rec.860", "rec.796", "rec.797", nullptr, nullptr, 1 },
  { "rec.layer3.10.bn1", "rec.799", "rec.800", "rec.861", "rec.802", "rec.803", nullptr, nullptr, 1 },
  { "rec.layer3.11.bn1", "rec.805", "rec.806", "rec.862", "rec.808", "rec.809", nullptr, nullptr, 1 },
  { "rec.layer3.12.bn1", "rec.811", "rec.812", "rec.863", "rec.814", "rec.815", nullptr, nullptr, 1 },
  { "rec.layer3.13.bn1", "rec.817", "rec.818", "rec.864", "rec.820", "rec.821", nullptr, nullptr, 1 },
  // layer4: 3 blocks
  { "rec.layer4.0.bn1", "rec.823", "rec.824", "rec.865", "rec.826", "rec.827", "rec.829", "rec.830", 2 },
  { "rec.layer4.1.bn1", "rec.832", "rec.833", "rec.866", "rec.835", "rec.836", nullptr, nullptr, 1 },
  { "rec.layer4.2.bn1", "rec.838", "rec.839", "rec.867", "rec.841", "rec.842", nullptr, nullptr, 1 },
};

// Offset of each stage's first block within kRecBlocks (prefix sum of counts).
constexpr int kRecStageOffset[kRecStageCount] = {0, 3, 7, 21};

// ArcFace IResNet50 "output_layer" head (insightface arcface_torch IResNet):
//   x = bn2(x)                  BatchNorm2d over the 512 trunk channels
//   x = flatten(x)              NCHW -> [N, 512*7*7] = [N, 25088]
//   x = fc(x)                   Linear 25088 -> 512  (rec.fc.weight ne [25088,512])
//   x = features(x)             BatchNorm1d over the 512 features (affine)
// followed by an L2-normalize to yield primary.normed_embedding. Verbatim GGUF
// names from tensor_manifest_buffalo_l.md; both BNs use the same 1e-5 eps.
constexpr const char* kOutBn2     = "rec.bn2";       // .weight/.bias/.running_*
constexpr const char* kFcW        = "rec.fc.weight"; // ggml ne [25088, 512] = [in, out]
constexpr const char* kFcB        = "rec.fc.bias";   // [512]
constexpr const char* kFeaturesBN = "rec.features";  // BatchNorm1d, .weight/.bias/.running_*

// BatchNorm1d folded as a per-feature affine over ne0 (the 512-d FC output),
// mirroring graph_ops batchnorm but broadcasting over the row dimension instead
// of the channel dimension. y = x * (gamma/sqrt(var+eps)) + (beta - mean*scale).
ggml_tensor* batchnorm1d(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                         const char* prefix, float eps,
                         std::vector<std::vector<float>>& keep) {
    (void)keep;  // affine is host-folded (bn_fold_params); no per-call eps leaf
    // Same host-constant fold as the conv-path batchnorm: scale/shift computed
    // once and cached, injected as two F32 leaves instead of rebuilding the
    // add1/sqrt/div/sub/mul chain every compute.
    const float* scale; const float* shift; int64_t F = 0;
    bn_fold_params(ml, prefix, eps, &scale, &shift, &F);
    const int64_t ne[4] = {F, 1, 1, 1};
    ggml_tensor* s = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne,
                                        scale, F * sizeof(float));
    ggml_tensor* b = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne,
                                        shift, F * sizeof(float));
    // x is [F, 1]; scale/shift [F] broadcast over the (single) row.
    return ggml_add(ctx, ggml_mul(ctx, x, s), b);
}
} // namespace

const int kRecBlockCounts[kRecStageCount] = {3, 4, 14, 3};

const RecBlock& rec_block(int stage, int idx) {
    FD_ASSERT(stage >= 0 && stage < kRecStageCount);
    FD_ASSERT(idx >= 0 && idx < kRecBlockCounts[stage]);
    return kRecBlocks[kRecStageOffset[stage] + idx];
}

ggml_tensor* arcface_stem(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                          std::vector<std::vector<float>>& keep) {
    (void)keep;  // reserved for host-side buffers used by later trunk stages (4.2/4.3)
    // SMALL-IC STEM: conv(3->64, k3, s1, p1) followed by a fused bias + per-channel
    // PReLU epilogue. The stem conv is memory-bound, not compute-bound: IC=3 is a
    // tiny contraction (43 MFLOP), so the stem's wall time is dominated by streaming
    // its 3.2 MB (112x112x64 f32) output, NOT by the conv FMAs. The previous
    // epilogue ran the BN-folded bias add and the PReLU as SEPARATE ggml ops
    // (ggml_add bias, then prelu = relu + sub + mul + add) -- five full-buffer
    // passes over that 3.2 MB. Folding them into ONE fused_prelu_bias pass
    // (v = x + bias[c]; v>0 ? v : alpha[c]*v) is the actual stem recovery: it
    // removes four 3.2 MB bandwidth passes for a memory-bound stage. The conv stays
    // on its routed kernel (Winograd / small-IC direct), which already broadcasts
    // the IC=3 input as scalars -- it is NOT strided through a wasteful 16-wide
    // (nChw16c) load, the trap a naive "put the stem in the blocked island" would
    // hit (reorder_in would pad IC 3->16 and read 3 useful lanes per 16-wide
    // access). Arithmetic order is unchanged, so embedding parity is preserved.
    x = conv2d(ctx, ml, x, kStemConvW, /*bias=*/nullptr, /*stride=*/1, /*pad=*/1, /*relu=*/false);
    ggml_tensor* bias  = clone_weight(ctx, ml, kStemConvB);   // folded-BN bias [64]
    ggml_tensor* alpha = clone_weight(ctx, ml, kStemPReluA);  // PReLU slope [1,1,64]
    return fused_prelu_bias(ctx, x, bias, alpha);
}

ggml_tensor* arcface_ir_block(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                              const RecBlock& n, std::vector<std::vector<float>>& keep) {
    ggml_tensor* identity = x;

    // Main path: bn1(x) -> conv1(k3 s1 p1) -> prelu -> conv2(k3 stride p1).
    // The pre-activation bn1 is an explicit BatchNorm; conv1/conv2 carry their
    // following BN (bn2/bn3) folded into the bias.
    ggml_tensor* y = batchnorm(ctx, ml, x, n.bn1, kRecBnEps, keep);
    y = conv2d(ctx, ml, y, n.conv1_w, n.conv1_b, /*stride=*/1, /*pad=*/1, /*relu=*/false);
    y = prelu(ctx, y, clone_weight(ctx, ml, n.prelu_a));
    y = conv2d(ctx, ml, y, n.conv2_w, n.conv2_b, /*stride=*/n.stride, /*pad=*/1, /*relu=*/false);

    // Downsample shortcut (first block of a stage): a 1x1 strided conv (bn folded)
    // on the block INPUT x, matching the residual's spatial/channel shape.
    if (n.ds_w) {
        identity = conv2d(ctx, ml, x, n.ds_w, n.ds_b, /*stride=*/n.stride, /*pad=*/0,
                          /*relu=*/false);
    }
    return ggml_add(ctx, y, identity);
}

namespace {
// Blocked pre-activation BatchNorm: same host-fold as graph_ops batchnorm (scale =
// gamma/sqrt(var+eps), shift = beta - mean*scale, cached once per model), injected
// as two F32 [C] leaves, applied per-channel over the 16-lane blocked buffer.
ggml_tensor* blocked_bn(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* xb,
                        const char* prefix, float eps) {
    const float* scale; const float* shift; int64_t C = 0;
    bn_fold_params(ml, prefix, eps, &scale, &shift, &C);
    const int64_t ne[4] = {C, 1, 1, 1};
    ggml_tensor* s = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne, scale, C * sizeof(float));
    ggml_tensor* b = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne, shift, C * sizeof(float));
    return blocked_scale_shift(ctx, xb, s, b);
}

// One IResNet IBasicBlock entirely on the blocked (nChw16c) buffer: the conv
// kh,kw,ic accumulation order matches the per-conv directconv (bit-identical, a
// layout-only change), the BN folds match graph_ops batchnorm, and the PReLU slope
// is the same per-channel rec.84x leaf. conv bias (folded bn2/bn3) is fused into
// the conv register tile; bn1 and PReLU are standalone blocked passes. No reorder
// in the block body -- the whole 24-block backbone is ONE island.
ggml_tensor* arcface_ir_block_blocked(ggml_context* ctx, const ModelLoader& ml,
                                      ggml_tensor* xb, const RecBlock& n) {
    ggml_tensor* identity = xb;
    ggml_tensor* y = blocked_bn(ctx, ml, xb, n.bn1, kRecBnEps);
    y = blocked_conv3x3(ctx, clone_weight(ctx, ml, n.conv1_w), y, /*pad=*/1, /*stride=*/1,
                        clone_weight(ctx, ml, n.conv1_b), /*do_relu=*/false);
    y = blocked_prelu(ctx, y, clone_weight(ctx, ml, n.prelu_a));
    y = blocked_conv3x3(ctx, clone_weight(ctx, ml, n.conv2_w), y, /*pad=*/1, /*stride=*/n.stride,
                        clone_weight(ctx, ml, n.conv2_b), /*do_relu=*/false);
    if (n.ds_w) {
        identity = blocked_conv1x1(ctx, clone_weight(ctx, ml, n.ds_w), xb, /*stride=*/n.stride,
                                   clone_weight(ctx, ml, n.ds_b), /*do_relu=*/false);
    }
    return blocked_add(ctx, y, identity);  // IResNet residual: no post-add activation
}

// How many leading IR blocks to run inside the blocked island.
// FACEDETECT_BLOCKED_BACKBONE override: "0"/"off" forces the per-conv path (the
// shape-gated directconv/Winograd via conv2d); "all"/"on" runs every block blocked
// (the whole backbone is ONE island, exactly 2 reorders); a positive integer runs
// that many leading blocks blocked (incremental island growth + re-gating). UNSET
// defaults to the whole backbone IFF the AVX-512 blocked fast path is available at
// runtime; on non-AVX512 hosts it defaults OFF so they keep the AVX2 per-conv path.
int blocked_island_blocks(int n_blocks) {
    const char* e = std::getenv("FACEDETECT_BLOCKED_BACKBONE");
    if (!e || !e[0]) return directconv_blocked_available() ? n_blocks : 0;
    if (!std::strcmp(e, "0") || !std::strcmp(e, "off")) return 0;
    if (!std::strcmp(e, "all") || !std::strcmp(e, "on")) return n_blocks;
    int v = std::atoi(e);
    if (v < 0) v = 0;
    if (v > n_blocks) v = n_blocks;
    return v;
}
} // namespace

std::vector<float> arcface_embed(const ModelLoader& ml, const Image& aligned112,
                                 Backend& be) {
    const int sz = (int)ml.config().rec_input_size;  // 112
    // The aligned crop is already RGB (norm_crop warps the RGB source), so
    // swap_rb=false yields R,G,B planes - identical to the reference's
    // blobFromImage(swapRB=True) on the BGR crop (the 4.1 stem gate convention).
    std::vector<float> blob = to_blob(aligned112, sz, 127.5f, 127.5f, /*swap_rb=*/false);

    // MobileFaceNet recognizer (buffalo_s / w600k_mbf): the backbone is NOT the
    // hand-mapped IResNet50 below, so replay its embedded ONNX topology through the
    // shared metadata-driven interpreter, then L2-normalize the raw 512-d output to
    // match insightface's normed_embedding (insightface normalizes outside ONNX).
    const FaceConfig& cfg = ml.config();
    if (!cfg.rec_graph.empty()) {
        std::vector<float> e = run_onnx_graph(ml, "rec.", cfg.rec_graph,
                                              cfg.rec_graph_output, cfg.rec_graph_input,
                                              blob, sz, be, kRecBnEps);
        double ss = 0.0;
        for (float v : e) ss += (double)v * v;
        const float inv = (float)(1.0 / std::sqrt(ss > 0.0 ? ss : 1.0));
        for (float& v : e) v *= inv;
        return e;
    }

    // keep owns host-side BN-eps scalar buffers; graph_input_tensor defers the
    // host->device copy until after gallocr alloc, so it MUST outlive compute.
    std::vector<std::vector<float>> keep;
    std::vector<float> out;
    const bool ok = be.compute([&](ggml_context* ctx) -> ggml_tensor* {
        const int64_t ne[4] = {sz, sz, 3, 1};
        ggml_tensor* x = graph_input_tensor(ctx, GGML_TYPE_F32, 4, ne,
                                            blob.data(), blob.size() * sizeof(float));
        // Stem -> 24 IR residual blocks across the 4 stages ([3,4,14,3]).
        x = arcface_stem(ctx, ml, x, keep);

        // Blocked-island backbone (nChw16c): keep the leading `kb` IR blocks in the
        // blocked layout between ONE reorder-in (here, after the stem) and ONE
        // reorder-out, amortizing the per-conv NCHW<->blocked round trip over the
        // whole body. kb == total makes the entire 24-block backbone a single island
        // with exactly 2 reorders (vs ~50 per-conv scatter/gather passes). Default is
        // the whole backbone on AVX-512 hosts (where the blocked fast path runs),
        // else 0 (per-conv directconv/Winograd via conv2d, as before).
        const int total_blocks = kRecBlockCounts[0] + kRecBlockCounts[1] +
                                 kRecBlockCounts[2] + kRecBlockCounts[3];
        const int kb = blocked_island_blocks(total_blocks);
        ggml_tensor* xb = (kb > 0) ? blocked_reorder_in(ctx, x) : nullptr;
        int flat = 0;
        for (int s = 0; s < kRecStageCount; ++s)
            for (int i = 0; i < kRecBlockCounts[s]; ++i, ++flat) {
                const RecBlock& blk = rec_block(s, i);
                if (flat < kb) {
                    xb = arcface_ir_block_blocked(ctx, ml, xb, blk);
                } else {
                    // First post-island block: reorder nChw16c -> NCHW once. The
                    // IResNet stage widths (64/128/256/512) are multiples of 16, so
                    // the true channel count is xb->ne[3]*16 (no padding lanes).
                    if (flat == kb && kb > 0)
                        x = blocked_reorder_out(ctx, xb, (int)xb->ne[3] * 16);
                    x = arcface_ir_block(ctx, ml, x, blk, keep);
                }
            }
        // Whole-backbone island: reorder out after the last blocked block.
        if (kb >= total_blocks && kb > 0)
            x = blocked_reorder_out(ctx, xb, (int)xb->ne[3] * 16);

        // Output head: bn2 -> flatten(NCHW) -> fc(25088->512) -> features BN.
        x = batchnorm(ctx, ml, x, kOutBn2, kRecBnEps, keep);
        // ggml [W,H,C,N] flattens W-fastest (== PyTorch NCHW flatten, w fastest),
        // so a plain reshape lands on the FC's expected 25088 input order.
        x = ggml_reshape_2d(ctx, ggml_cont(ctx, x), (int64_t)ggml_nelements(x), 1);
        ggml_tensor* W = clone_weight(ctx, ml, kFcW);   // ne [25088, 512] = [in, out]
        x = ggml_mul_mat(ctx, W, x);                    // -> [512, 1]
        x = ggml_add(ctx, x, clone_weight(ctx, ml, kFcB));
        x = batchnorm1d(ctx, ml, x, kFeaturesBN, kRecBnEps, keep);

        // L2-normalize over the 512 features (ne0): x / sqrt(sum(x^2)).
        ggml_tensor* ss  = ggml_sum_rows(ctx, ggml_sqr(ctx, x));  // [1, 1]
        ggml_tensor* nrm = ggml_sqrt(ctx, ss);
        return ggml_div(ctx, x, nrm);                            // broadcast over ne0
    }, out);
    if (!ok) throw std::runtime_error("facedetect: arcface graph compute failed");
    return out;
}

} // namespace fd
