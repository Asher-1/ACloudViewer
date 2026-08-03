#pragma once
#include <string>
#include <vector>
#include "image_io.hpp"

struct ggml_context;
struct ggml_tensor;

namespace fd {

class ModelLoader;
class Backend;
class Model;
struct Detection;

// Metadata-driven ONNX forward-graph interpreter, shared by the MiniFASNet
// anti-spoof ensemble and the MobileFaceNet (w600k_mbf) recognizer. Replays the
// embedded node specs ("op;out;in0,in1,...;k=v,k=v") against `prefix`-namespaced
// GGUF weights: each node input resolves to a prior activation or a weight tensor.
// The 4-D input blob (NCHW f32, ggml ne [size,size,3,1]) is seeded under both
// "input" and the graph's real ONNX input name. Returns the raw `out_name` tensor
// (callers apply any post-processing, e.g. the recognizer's L2-normalize). Supports
// Conv (incl. depthwise via group), PRelu, Relu, Sigmoid, Add, Sub, Mul, Dropout
// (eval identity), GlobalAveragePool, Flatten, MatMul, Gemm (transB), and
// BatchNormalization (both 1-D linear and 4-D spatial, with a per-node `e=` epsilon).
// The SFace recognizer (sface_graph.cpp) drives the Sub/Dropout/spatial-BN path.
std::vector<float> run_onnx_graph(const ModelLoader& ml, const std::string& prefix,
                                  const std::vector<std::string>& specs,
                                  const std::string& out_name,
                                  const std::string& input_name,
                                  const std::vector<float>& blob, int size,
                                  Backend& be, float bn_eps);

// Multi-output variant of the interpreter, used by the SCRFD det_2.5g / det_500m
// detectors (buffalo_m / buffalo_s): replays the embedded `facedetect.detector.graph`
// node specs (input/attr separator '|', since head weight names embed commas) and
// returns the contents of every tensor in `output_names`, in that order. For SCRFD
// the 9 outputs are the per-stride heads in ONNX graph-output order: scores
// s8/16/32 (post-Sigmoid), bbox s8/16/32, kps s8/16/32, each already flattened to
// the host decode's (H*W*num_anchors, C) row layout by the head Transpose+Reshape.
// Adds MaxPool / AveragePool / Resize (2x nearest) / Transpose / Reshape to the
// op set on top of run_onnx_graph's. `bn_eps` is unused unless the graph carries a
// standalone BatchNormalization (the SCRFD exports fold BN into conv).
std::vector<std::vector<float>> run_onnx_graph_multi(
    const ModelLoader& ml, const std::string& prefix,
    const std::vector<std::string>& specs, const std::string& input_name,
    const std::vector<std::string>& output_names,
    const std::vector<float>& blob, int size, Backend& be, float bn_eps);

// MiniFASNet anti-spoof ensemble (Silent-Face), the `as<i>.` GGUF prefix.
//
// The buffalo anti-spoof pack is a two-model ensemble: as0 = MiniFASNetV2 at
// crop scale 2.7, as1 = MiniFASNetV1SE at scale 4.0. Each is an 80x80 classifier
// (3 logits, softmax index 1 = "real"). Both are MobileFaceNet-style nets (a
// stride-2 stem, depthwise-separable residual blocks across three stages, then a
// 5x5 depthwise pool, a 512->128 linear, BatchNorm1d, and a 128->3 head); V1SE
// additionally carries Squeeze-and-Excitation blocks at the last block of each
// stage. Rather than hard-code the two pruned architectures, the C++ engine
// REPLAYS the ONNX node topology embedded in the GGUF (FaceConfig::antispoof_graphs)
// against the per-model weights, mirroring the reference 1:1.
//
// Preprocessing is parity-critical: the reference feeds the raw BGR crop straight
// in (np.transpose, no /255, no mean/std, no swapRB), so the network sees B,G,R
// planes. fd::Image is RGB, so to_blob(..., swap_rb=true) lands on those B,G,R
// planes (the same blob the reference fed onnxruntime).

// Run a single ensemble member's MiniFASNet graph on a preprocessed 80x80 BGR
// blob (NCHW f32, B/G/R planes) and return the raw 3 logits. `model_idx` selects
// the ensemble member (the `as<i>.` weights + the i-th embedded topology).
// Exposed so a parity test can gate the graph numerics on the reference's exact
// 80x80 crop, isolated from the host-side bbox crop+resize.
std::vector<float> minifasnet_forward(const ModelLoader& ml, int model_idx,
                                      const std::vector<float>& blob, int size,
                                      Backend& be);

// Reproduce the reference `_crop_face`: expand the detection box about its center
// by `scale` (clamped so the crop fits inside the image), take the integer-bounded
// axis-aligned crop, and resize to `size`x`size` (cv2.resize INTER_LINEAR, via the
// shared fd::warp_affine). Fills `out` (size x size RGB).
bool antispoof_crop(const Image& img, const Detection& d, float scale, Image& out,
                    int size);

// Averaged "real"-class probability over the ensemble: for each member, take its
// scale-specific crop, run the graph, softmax, and average index 1 across members
// (the same ensembling as upstream test.py). Returns accum[1] in [0,1].
float antispoof_real_prob(const Model& m, const Image& img, const Detection& d);

// Ensemble score per the verify-veto contract: the averaged real probability when
// argmax of the averaged softmax is the "real" class (index 1), else 0.0. A face
// is judged live when this is >= 0.5 (the reference threshold).
float antispoof_score(const Model& m, const Image& img, const Detection& d);

} // namespace fd
