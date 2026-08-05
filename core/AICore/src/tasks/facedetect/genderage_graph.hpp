#pragma once
#include <utility>
#include <vector>
#include "image_io.hpp"

struct ggml_context;
struct ggml_tensor;

namespace fd {

class ModelLoader;
class Backend;
struct Detection;

// insightface genderage head (buffalo_l `genderage.onnx`, the `ga.` GGUF prefix).
//
// A small MobileNet-0.25 attribute net: a stride-2 stem conv (3->16) followed by
// 11 depthwise-separable blocks (conv_2..conv_12), then the trunk splits into two
// task branches that both consume conv_12's output:
//   * t0 (gender): conv_13_t0 / conv_14_t0 -> GlobalAvgPool -> FC 256->2 logits
//   * t1 (age)   : conv_13_t1 / conv_14_t1 -> GlobalAvgPool -> FC 256->1 scalar
// concatenated to the ONNX `fc1` [g0, g1, age]. Unlike the ArcFace export the
// BatchNorms are NOT folded into the convs here: every conv is bias-less and
// followed by an explicit BatchNormalization (gamma/beta/moving_mean/moving_var)
// then ReLU. The net also carries its own input normalization as the first two
// ONNX nodes: y = (x - scalar_op1) * scalar_op2 == (x - 127.5)/128, so the host
// blob is the raw [0,255] RGB crop (mean 0, std 1). BN epsilon is 1e-3 (mxnet).

// Run the genderage graph on an already-aligned 96x96 RGB crop and return the
// raw 3-float `fc1` output [gender_logit0, gender_logit1, age_raw]. Exposed so a
// parity test can gate the graph numerics on the reference's exact crop pixels,
// isolated from the host-side bbox warp (mirroring the ArcFace isolated gate).
std::vector<float> genderage_forward(const ModelLoader& ml, const Image& crop96,
                                     Backend& be);

// Full genderage for one detected face: expand the detection box about its center
// by 1.5x, scale-fit to the model's square input (96), OpenCV-faithful warpAffine
// (face_align.transform), then run the graph and decode. Returns
// {gender, age} with gender 0=Woman / 1=Man (argmax of the two logits) and age in
// years (round(age_raw * 100)).
std::pair<int, int> genderage(const ModelLoader& ml, const Image& img,
                              const Detection& d, Backend& be);

} // namespace fd
