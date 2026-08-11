#pragma once
#include <vector>
#include "image_io.hpp"

struct ggml_context;
struct ggml_tensor;

namespace fd {

class ModelLoader;
class Backend;

// ArcFace (insightface w600k_r50, IResNet50, 512-d) recognizer graph.
//
// Input is the 112x112 aligned RGB crop produced by fd::norm_crop, normalized
// (x-127.5)/127.5 with R,G,B plane order (== insightface's
// cv2.dnn.blobFromImage(aligned_bgr, 1/127.5, (127.5,)*3, swapRB=True); our crop
// is already RGB, so to_blob uses swap_rb=false to land on the same R,G,B
// planes - the SAME convention as the SCRFD detector blob).
//
// Stem (this task, gated vs the reference's stem PReLU output): the first
// conv(3->64, k3, s1, p1) with the following BatchNorm folded into its bias
// (rec.685 weight / rec.686 bias), then a per-channel PReLU (slope rec.843).
// The IResNet IBasicBlock's own `bn1` is a PRE-activation BN on the block input
// (it cannot fold into a preceding conv, so it survives as an explicit
// rec.layerX.Y.bn1.* tensor); only the stem's post-conv BN folds. Returns the
// stem PReLU output [112,112,64,1] (ggml ne), matching ONNX value `479`.
ggml_tensor* arcface_stem(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                          std::vector<std::vector<float>>& keep);

// One IResNet IBasicBlock (insightface arcface_torch IR block). Verbatim GGUF
// tensor names for a single residual block. The ONNX export keeps each block's
// pre-activation `bn1` as an explicit BatchNorm node (semantic name), while the
// two inner convs have their following BN (bn2 / bn3) folded into a bias, so they
// appear under auto-numbered names. The block computes (confirmed node-by-node
// from w600k_r50.onnx, layer1.0 == nodes 2..7):
//   identity = x
//   y = bn1(x)                              [pre-activation BatchNorm]
//   y = conv1(y)         k3 s1 p1           [conv1_w / conv1_b, bn2 folded]
//   y = prelu(y)                            [prelu_a]
//   y = conv2(y)         k3 sStride p1      [conv2_w / conv2_b, bn3 folded]
//   if downsample: identity = conv_ds(x)    k1 sStride p0   [ds_w / ds_b, bn folded]
//   return y + identity
// The downsample shortcut conv runs on the block INPUT x (ONNX feeds it value
// '479', not the bn1 output), present only for the first block of each stage
// (stride 2 + channel change). `stride` is 2 for a stage's first block, else 1.
struct RecBlock {
    const char* bn1;      // bn1 prefix, e.g. "rec.layer1.0.bn1" (.weight/.bias/...)
    const char* conv1_w;  // conv1 weight (k3 s1 p1), bn2 folded into conv1_b
    const char* conv1_b;
    const char* prelu_a;  // per-channel PReLU slope
    const char* conv2_w;  // conv2 weight (k3 stride p1), bn3 folded into conv2_b
    const char* conv2_b;
    const char* ds_w;     // downsample conv weight (k1 stride p0) or nullptr
    const char* ds_b;     // downsample bias (bn folded) or nullptr
    int stride;           // conv2 + downsample stride (2 for a stage's first block)
};

// IResNet50 trunk layout: [3,4,14,3] IBasicBlocks across 4 stages, 24 total. The
// verbatim per-block tensor names (extracted from w600k_r50.onnx) so the 4.3
// stage loop can iterate blocks; Task 4.2 gates kRecStages[0][0] (layer1.0).
constexpr int kRecStageCount = 4;
extern const int kRecBlockCounts[kRecStageCount];          // {3,4,14,3}
const RecBlock& rec_block(int stage, int idx);             // bounds-checked accessor

// One IR residual block as a ggml graph (see RecBlock for the exact structure).
// `x` is the block input [W,H,C,1] (ggml ne); returns the residual sum. `keep`
// owns host-side scalar buffers (BN eps) registered as graph inputs.
ggml_tensor* arcface_ir_block(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                              const RecBlock& n, std::vector<std::vector<float>>& keep);

// Full L2-normalized 512-d ArcFace embedding for an aligned 112x112 RGB crop.
// Completed in Task 4.3 (residual trunk + bn2 + fc + features BN + L2 norm);
// declared here so downstream code can link against the final interface.
std::vector<float> arcface_embed(const ModelLoader& ml, const Image& aligned112,
                                 Backend& be);

} // namespace fd
