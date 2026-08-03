#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
struct ggml_tensor;
struct ggml_context;
struct gguf_context;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
namespace fd {

// Parsed config for a face-recognition model pack. All values come from GGUF KV
// (metadata-driven). Defaults are insightface buffalo_l conventions; absent keys
// keep the default so a minimal pack still loads.
struct FaceConfig {
    std::string arch;                 // e.g. "scrfd+arcface", "yunet+sface"

    // detector (SCRFD / YuNet)
    std::string detector;             // "scrfd" (default) or "yunet"
    uint32_t det_input_size = 640;    // square detector input (letterboxed)
    std::vector<int32_t> det_strides; // FPN strides, e.g. [8,16,32]
    uint32_t det_num_anchors = 2;     // anchors per location (SCRFD: 2)
    float det_score_thresh = 0.5f;    // detection confidence threshold
    float det_nms_thresh   = 0.4f;    // NMS IoU threshold
    // Optional embedded SCRFD detector ONNX topology (buffalo_m det_2.5g /
    // buffalo_s det_500m). When non-empty the detector is NOT the hand-mapped
    // det_10g graph: the C++ replays these node specs ("op;out;in0|in1|...;k=v|k=v")
    // through the shared graph interpreter, resolving `det.`-prefixed weights, and
    // feeds the 9 named per-stride heads (det_graph_outputs, in the order scores
    // s8/16/32, bbox s8/16/32, kps s8/16/32) into the existing anchor decode + NMS.
    // Empty for buffalo_l, which keeps the hand-mapped det_10g path.
    std::vector<std::string> det_graph;
    std::string det_graph_input;
    std::vector<std::string> det_graph_outputs;

    // recognizer (ArcFace / SFace)
    std::string recognizer;           // "arcface" (default) or "sface"
    uint32_t rec_input_size = 112;    // aligned crop size (ArcFace: 112)
    uint32_t embed_dim      = 512;    // embedding dimension (ArcFace: 512, SFace: 128)
    float verify_threshold  = 0.35f;  // default cosine-distance match threshold
    // Optional recognizer ONNX node topology (buffalo_s MobileFaceNet / w600k_mbf).
    // When non-empty the recognizer is NOT the hand-mapped IResNet50 graph: the C++
    // replays these node specs ("op;out;in0,in1,...;k=v,k=v") through the shared
    // graph interpreter, resolving `rec.`-prefixed weights, then L2-normalizes the
    // raw output. rec_graph_input / rec_graph_output are the ONNX graph IO names.
    std::vector<std::string> rec_graph;
    std::string rec_graph_input;
    std::string rec_graph_output;

    // genderage head (optional)
    bool genderage_present = false;
    uint32_t genderage_input_size = 96;

    // MiniFASNet anti-spoof ensemble (optional). The buffalo anti-spoof pack is
    // V2@scale2.7 + V1SE@scale4.0 at 80x80; the scales/sizes live in KV so the
    // host warp matches the reference exactly.
    bool antispoof_present = false;
    uint32_t antispoof_input_size = 80;
    std::vector<float> antispoof_scales; // e.g. [2.7, 4.0]
    // Per-model ONNX node topology (one entry per ensemble member, aligned with
    // antispoof_scales). Each inner vector is the list of node specs
    // ("op;out;in0,in1,...;k=v,k=v") the C++ interpreter replays; the matching
    // graph output tensor name is in antispoof_graph_outputs. Empty when absent.
    std::vector<std::vector<std::string>> antispoof_graphs;
    std::vector<std::string> antispoof_graph_outputs;

    // Dense-landmark heads (insightface 2d106det / 1k3d68), engine-level only: NO
    // LocalAI proto RPC / API consumes them yet (the Detect RPC returns just the
    // 5-point SCRFD kps). Each head is a 192x192 regressor whose ONNX topology is
    // embedded as `graph` and replayed by the shared interpreter against the
    // l2d./l3d.-prefixed weights. `dim` is 2 (106-pt) or 3 (68-pt); `num_points`
    // selects the LAST N decoded rows (1k3d68 emits 1103 rows, of which 68 are the
    // landmarks). Both heads use input_mean 0 / input_std 1 (raw [0,255] pixels):
    // 2d106det carries (x-127.5)/128 in-graph, 1k3d68 absorbs the input scale into
    // its leading bn_data BatchNorm. `present` is false when the pack has no head.
    struct LandmarkHead {
        bool present = false;
        uint32_t input_size = 192;
        uint32_t num_points = 0;
        uint32_t dim = 0;
        float input_mean = 0.0f;
        float input_std = 1.0f;
        std::vector<std::string> graph;
        std::string graph_input = "data";
        std::string graph_output = "fc1";
    };
    LandmarkHead landmark_2d;   // l2d.* : 106-point 2D
    LandmarkHead landmark_3d;   // l3d.* : 68-point 3D
};

class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader();
    bool load(const std::string& path);
    const FaceConfig& config() const { return cfg_; }
    ggml_tensor* tensor(const std::string& name) const; // nullptr if absent
    ggml_context* ggml_ctx() const { return ctx_; }

    // Give every weight tensor a backend buffer (ONCE) so graphs can reference
    // the loader's tensors DIRECTLY as leaves with zero per-call copying. Mirrors
    // parakeet.cpp's ModelLoader::realize_weights (CPU zero-copy from_ptr path;
    // device path uploads into a backend buffer). Returns false on failure.
    bool realize_weights(ggml_backend_t backend);
    bool weights_realized() const { return weights_buf_ != nullptr; }
private:
    FaceConfig cfg_;
    gguf_context* gguf_ = nullptr;
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer_t weights_buf_ = nullptr;
    ggml_context* device_ctx_ = nullptr;  // no_alloc mirror ctx for device weights
    std::unordered_map<std::string, ggml_tensor*> tensors_;
};
}
