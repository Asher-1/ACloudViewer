#pragma once
#include <array>
#include <vector>

#include "image_io.hpp"

namespace fd {

class ModelLoader;
class Backend;
struct Detection;

// One dense-landmark point in image (or crop) space. `z` is 0 for the 2D head and
// the (scale-corrected) depth for the 3D head.
struct LandmarkPoint {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

// ENGINE-LEVEL dense-landmark capability (insightface 2d106det / 1k3d68).
//
// There is NO LocalAI proto RPC / API endpoint that consumes dense landmarks yet:
// the Detect RPC returns only the 5-point SCRFD kps. These heads are reachable from
// the C++ engine + CLI (facedetect-cli landmarks) + parity gates ONLY. Exposing them
// through LocalAI needs a new dense-landmark RPC + endpoint + capability registry.
//
// The heads ride the SAME metadata-driven ONNX-graph interpreter as the MobileFaceNet
// recognizer / MiniFASNet ensemble (run_onnx_graph in antispoof_graph.cpp): the GGUF
// carries the topology verbatim and the per-head `l2d.` / `l3d.` weights.

// Build the insightface Landmark.get crop: scale-about-center the detection box
// (expanded 1.5x) to fit the square `size` model input, centered (face_align.transform
// with rotation 0). Returns the forward affine M (src -> crop) in `M`; fills `out`
// with the size x size RGB crop. Identical geometry to the genderage head's crop,
// only the size differs (192 vs 96). Returns false on warp failure.
bool landmark_crop(const Image& img, const Detection& d, int size,
                   std::array<float, 6>& M, Image& out);

// Run one landmark head's ONNX graph on an already-cropped size x size RGB face and
// return the RAW `fc1` output (212 for 2d106det, 3309 for 1k3d68). `three_d` selects
// the head; throws if that head is absent from the pack.
std::vector<float> landmark_forward(const ModelLoader& ml, bool three_d,
                                    const Image& crop, Backend& be);

// Decode a raw `fc1` output to CROP-space points (insightface Landmark.get, minus the
// inverse-affine step): reshape to (-1, dim), take the LAST `num_points` rows, add 1
// to x,y and scale x,y (and z for the 3D head) by input_size/2. `three_d` selects the
// head geometry from the pack config.
std::vector<LandmarkPoint> landmark_decode_crop(const ModelLoader& ml, bool three_d,
                                                const std::vector<float>& raw);

// Map CROP-space points back to image space via the inverse of the forward affine M
// (insightface trans_points): x,y by the full 2x3 inverse, z (3D) scaled by the
// inverse's uniform scale sqrt(IM00^2 + IM01^2).
std::vector<LandmarkPoint> landmark_to_image(const std::vector<LandmarkPoint>& pts,
                                             const std::array<float, 6>& M);

// End-to-end: crop the detection box, run the head, decode, and map back to image
// space. `three_d` selects the 68-point 3D head over the 106-point 2D head.
std::vector<LandmarkPoint> landmarks(const ModelLoader& ml, bool three_d,
                                     const Image& img, const Detection& d, Backend& be);

} // namespace fd
