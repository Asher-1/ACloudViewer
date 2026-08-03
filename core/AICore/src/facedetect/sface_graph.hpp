#pragma once
#include <vector>
#include "image_io.hpp"

namespace fd {

class ModelLoader;
class Backend;

// SFace recognizer (OpenCV-Zoo `face_recognition_sface_2021dec`, Apache-2.0): the
// commercial-friendly 128-d alternative to the non-commercial insightface ArcFace,
// paired with the YuNet detector. It is a MobileFaceNet-style depthwise-separable
// CNN whose ONNX carries its OWN input normalization in-graph (a `Sub`(127.5) then
// `Mul`(1/128) before the stem conv) and tags every BatchNormalization with its own
// epsilon (1e-3 conv BNs, 2e-5 for bn1/fc1). Rather than hand-map it, the C++ engine
// REPLAYS the embedded ONNX topology (`facedetect.recognizer.graph`) through the
// shared metadata-driven interpreter (run_onnx_graph), the same path as the
// MobileFaceNet (w600k_mbf) recognizer - it adds only the `Sub`/`Dropout` ops and the
// spatial (4-D) BatchNorm + per-node epsilon to that interpreter.

// Raw 128-d SFace feature: the network's `fc1` BatchNorm output, NOT normalized.
// Matches OpenCV `cv2.FaceRecognizerSF.feature` exactly. The aligned crop is the
// 112x112 RGB SFace `alignCrop` output; the (x-127.5)/128 normalization happens
// INSIDE the graph, so the blob is raw pixels (mean 0, std 1). cv2's feature uses
// blobFromImage(swapRB=true) on its BGR crop, so the net sees R,G,B; our crop is
// already RGB, so the blob is built with NO channel swap.
std::vector<float> sface_feature(const ModelLoader& ml, const Image& aligned112,
                                 Backend& be);

// L2-normalized 128-d embedding: sface_feature / ||sface_feature||. This is the
// direction OpenCV's cosine match (FR_COSINE) compares; pairs with the YuNet
// detector to form the fully Apache-2.0 recognition pipeline.
std::vector<float> sface_embed(const ModelLoader& ml, const Image& aligned112,
                               Backend& be);

} // namespace fd
