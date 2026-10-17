// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// GKDT ggml runtime - session and graph builders (in-tree port of upstream
// cpp_ggml/src/gkd_graph.*).
//
// The PyTorch GKDModel is decomposed into three ggml graphs that map 1:1 onto
// the official forward pass (gkd_inference_lib/gkd_model.py):
//
//   vision_graph : images  {384,384,3,B}      -> DINOv3 patch tokens {D, 576, B}
//   text_graph   : token ids {77, T}          -> adapted text features {2048, 77, T}
//   detect_graph : qf {24,24,C,1}, prompts    -> fused heatmaps {N, 9216}
//                  (KGTransformer + DetectionHead + openkd_heatmap_fuse)
//
// Host-side steps (tiny, kept outside the graphs on purpose): BPE tokenization,
// EOT argmax pooling + lower-half slice, soft-fiber-Gaussian visual prompt
// pooling, prompt padding (official PAD_KPS protocol) and heatmap decode.
//
// All tensors use the ggml convention ne[0] fastest. Vision/text streams keep
// the [D, T] "channels-fastest" layout so LayerNorm, channel norms and the 1x1
// kernel conv are natural row operations.
#pragma once

#include "tasks/gkd/gkd_backend.hpp"
#include "tasks/gkd/gkd_gguf_loader.hpp"
#include "tasks/gkd/gkd_image_io.hpp"
#include "tasks/gkd/gkd_tokenizer.hpp"

#include <memory>
#include <string>
#include <vector>

namespace gkd {

struct DetectInput {
    // Query ROI images (already preprocessed to img_size x img_size, RGB).
    std::vector<RgbImage> query_images;
    // Optional 1-shot support image with keypoints in ORIGINAL image coords.
    RgbImage support_image;
    bool has_support = false;
    std::vector<float> support_kps_xy;    // N_v x 2
    std::vector<uint8_t> support_kps_vis; // N_v
    // Text prompts (one keypoint name per prompt).
    std::vector<std::string> kps_texts;
};

struct DetectOutput {
    int n_prompts = 0;            // N_t + N_v (unpadded)
    std::vector<float> kps_norm;  // N x 2 in -1..1 (input space)
    std::vector<float> scores;    // N
    std::vector<float> heatmaps;  // fused, N x heat_w x heat_w (per query 0)
    // ROI transform of the query image (source-pixel recovery for kps_norm).
    ScaleTrans trans;
};

// Steady-state per-stage latency (milliseconds).
struct StageTiming {
    double preprocess = 0;
    double vision = 0;
    double text = 0;
    double prompt_prep = 0;
    double detect = 0;
    double decode = 0;
    double total = 0;
};

class GkdSession {
public:
    ~GkdSession();

    static std::unique_ptr<GkdSession> create(const std::string& gguf_path,
                                              int n_threads,
                                              const std::string& device);

    const ModelParams& params() const { return model_->P; }
    const char* backend() const { return backend_name(bctx_); }
    int threads() const { return bctx_.n_threads; }

    // Full forward. `bbox` is (x1,y1,x2,y2) in original image coords; empty
    // means the whole image. Returns per-ROI keypoints/scores.
    bool detect(const RgbImage& image,
                const float* bboxes,
                int n_bbox,
                const DetectInput& extra,
                std::vector<DetectOutput>& out,
                StageTiming* timing = nullptr);

    // Parity hooks ----------------------------------------------------------
    // Dump internal tensors matching the official dump_taps.py. Call after
    // detect(); empty dir disables dumping.
    void set_dump_dir(const std::string& dir) { dump_dir_ = dir; }

private:
    GkdSession() = default;

    bool build_vision_graph(int n_images);
    bool build_text_graph(int n_texts);
    bool build_detect_graph(int n_t_pad, int n_v_pad);

    // host-side helpers
    void compute_rope_tables();
    void gaussian_pooling(const float* support_feats /*C x 576*/,
                          int n_kps,
                          const float* kps /*n x 2*/,
                          const uint8_t* vis,
                          float* out /*n x C*/) const;
    void decode_heatmaps(const float* fused /*{N_pad, HW} ggml: (j,pix) at j+pix*N_pad*/,
                         int n_pad,
                         int n_valid,
                         int heat_w,
                         const ScaleTrans& trans,
                         DetectOutput& out) const;

    void dump_tap(const char* name,
                  const std::vector<int64_t>& shape,
                  const float* data) const;

    std::unique_ptr<GkdModel> model_;
    BackendCtx bctx_{};
    SimpleTokenizer tokenizer_;
    bool tokenizer_ok_ = false;

    // static graph constants (reside in a backend buffer alongside the weights)
    ggml_context* static_ctx_ = nullptr;
    ggml_backend_buffer_t static_buf_ = nullptr;
    ggml_tensor* rope_sin_ = nullptr;        // {dh, 576}
    ggml_tensor* rope_cos_ = nullptr;        // {dh, 576}
    ggml_tensor* txt_causal_mask_ = nullptr; // {77, 77} f16

    // cached graphs
    struct VisionCache {
        int n_images = -1;
        int out_tokens = 576;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* input = nullptr;
        ggml_tensor* output = nullptr;
        ggml_context* ctx = nullptr;
        ggml_gallocr_t galloc =
                nullptr; // one gallocr per graph (shared gallocrs break on
        ~VisionCache();  // graph changes)
    } vcache_;
    struct TextCache {
        int n_texts = -1;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* input = nullptr;
        ggml_tensor* output = nullptr;
        ggml_context* ctx = nullptr;
        ggml_gallocr_t galloc = nullptr;
        ~TextCache();
    } tcache_;
    struct DetectCache {
        int n_t_pad = -1;
        int n_v_pad = -1;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* qf = nullptr;      // {fw, fw, D, 1} NHWC
        ggml_tensor* prompts = nullptr; // {D, N}
        ggml_tensor* mask = nullptr;    // {1, N}
        ggml_tensor* output = nullptr;  // {N, heat_hw} fused
        ggml_gallocr_t galloc = nullptr;
        ggml_context* ctx = nullptr;
        ~DetectCache();
    } dcache_;

    std::string dump_dir_;
};

}  // namespace gkd
