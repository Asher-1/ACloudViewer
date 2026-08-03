#include "sface_graph.hpp"
#include "antispoof_graph.hpp"   // run_onnx_graph (shared ONNX-graph interpreter)
#include "preprocess.hpp"
#include "model_loader.hpp"
#include "common.hpp"

#include <cmath>
#include <stdexcept>

namespace fd {

// Fallback BatchNormalization epsilon for any node whose spec omits one. The SFace
// export tags EVERY BatchNormalization with its epsilon (the converter emits `e=`),
// so this is never actually consumed - it only keeps run_onnx_graph's signature
// satisfied for a hypothetical untagged BN.
static constexpr float kSfaceBnEps = 1e-5f;

std::vector<float> sface_feature(const ModelLoader& ml, const Image& aligned112,
                                 Backend& be) {
    const FaceConfig& cfg = ml.config();
    if (cfg.rec_graph.empty())
        throw std::runtime_error("facedetect: SFace pack carries no recognizer graph");
    const int sz = (int)cfg.rec_input_size;  // 112

    // SFace normalizes IN-graph ((data - 127.5) * 0.0078125 via the leading Sub/Mul
    // nodes), so the blob is RAW pixels (mean 0, std 1). cv2.FaceRecognizerSF.feature
    // feeds blobFromImage with swapRB=TRUE on its BGR alignCrop, so the network sees
    // R,G,B planes; our aligned crop is already RGB, so swap_rb=false yields the same
    // R,G,B order (the identical convention to arcface_embed's RGB-crop path).
    std::vector<float> blob = to_blob(aligned112, sz, 0.0f, 1.0f, /*swap_rb=*/false);
    return run_onnx_graph(ml, "rec.", cfg.rec_graph, cfg.rec_graph_output,
                          cfg.rec_graph_input, blob, sz, be, kSfaceBnEps);
}

std::vector<float> sface_embed(const ModelLoader& ml, const Image& aligned112,
                               Backend& be) {
    std::vector<float> e = sface_feature(ml, aligned112, be);
    double ss = 0.0;
    for (float v : e) ss += (double)v * v;
    const float inv = (float)(1.0 / std::sqrt(ss > 0.0 ? ss : 1.0));
    for (float& v : e) v *= inv;
    return e;
}

} // namespace fd
