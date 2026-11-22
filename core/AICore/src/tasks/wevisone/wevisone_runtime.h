// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "aicore/image_view.h"
#include "aicore/pipeline_timing.h"
#include "common/ggml_backend_registry.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "tasks/wevisone/wevisone_gguf_loader.h"
#include "tasks/wevisone/wevisone_image_prep.h"
#include "tasks/wevisone/wevisone_mask_decoder_graph.h"
#include "tasks/wevisone/wevisone_types.h"

struct llama_model;
struct llama_context;
struct ggml_backend;
struct mtmd_context;
struct mtmd_input_chunks;

namespace wwv {

struct RuntimeOptions {
    std::string device = "auto";    // auto | cpu | cuda | vulkan [:N]
    int32_t threads = 0;            // <= 0: backend default
    std::string size = "2b";        // checkpoint family: 2b | 4b
    std::string precision = "f16";  // vlm gguf tier
    std::string decoder_precision = "f32";
    std::string models_dir;    // empty: shared wevisone cache dir
    std::string vlm_gguf;      // explicit override
    std::string mmproj_gguf;   // explicit override (published f16)
    std::string stages_dir;    // decoder-parity harness input stages
    bool no_taps = false;      // dump without the per-stage cpy taps
    std::string taps_filter;   // bisect probe: only matching taps
    std::string decoder_gguf;  // explicit override
    std::string mkl_log_lib;   // libtorch path for bit-exact log
    int32_t max_new_tokens = 4096;
    bool decode_masks = true;
    int32_t log_level = 1;
    std::string dump_dir;  // parity dumps ("" disables)
};

struct RunResult {
    std::string answer;
    std::vector<ResultBox> boxes;
    std::vector<bool> has_obj;
    bool masks_decoded = false;
    int32_t matte_width = 0;
    int32_t matte_height = 0;
    std::vector<float> matte;  // first matte of the run (w*h, row-major)
    int32_t prompt_tokens = 0;
    int32_t completion_tokens = 0;
    aicore_pipeline_timings timings{};
};

class WevisoneRuntime {
public:
    static std::unique_ptr<WevisoneRuntime> create(const RuntimeOptions& opts,
                                                   std::string* err);
    ~WevisoneRuntime();

    bool is_ready() const { return ready_; }
    const char* last_error() const { return last_error_.c_str(); }
    const std::string& resolved_device() const { return device_; }
    int32_t resolved_threads() const { return threads_; }
    const std::string& resolved_precision() const { return precision_; }

    // One typed request. Returns false with last_error() set on failure.
    // prompt_override (when non-empty) replaces the build_prompt output —
    // the evaluation protocol prompts (TMPL_MATTE single/instances) are
    // NOT build_prompt outputs, so the eval replication needs raw access.
    bool run_image(const aicore_image_view& img,
                   const std::string& task,
                   const std::string& query,
                   const float* ref_boxes,
                   int32_t n_ref_boxes,
                   RunResult* out,
                   const std::string& prompt_override = "");

    // engine.decode_masks contract: teacher-force `prompt` + `response`
    // (NO LLM generation) and decode mattes from the <OBJ> hidden states.
    // boxes01 (normalized xyxy) conditions the decoder on those UNROUNDED
    // values — the evaluation GT protocol (inference_matting.py
    // box_source=gt: the synthetic response carries the ROUNDED box while
    // the decoder conditions on the unrounded one); when boxes01 is null
    // the boxes are parsed from the response tokens (rounded, the
    // generated-path semantics). Query count =
    // min(n_boxes|parsed, <OBJ> count), the python truncation contract.
    bool run_decode_masks(const aicore_image_view& img,
                          const std::string& prompt,
                          const std::string& response,
                          const float* boxes01,
                          int32_t n_boxes,
                          RunResult* out);

    // Builds the task prompt exactly like prompts.py.
    static std::string build_prompt(const std::string& task,
                                    const std::string& query,
                                    const float* ref_boxes,
                                    int32_t n_ref_boxes,
                                    int32_t width,
                                    int32_t height);

    // Device admission check (models + graph workspace fit).
    bool warmup_backend();

private:
    WevisoneRuntime() = default;
    bool load_models(std::string* err);
    // Shared lease-registry device resolution (vlx_seek pattern): fills
    // gpu_lease_/cpu_lease_, device_ (resolved name) and cpu_mode_.
    bool resolve_device(std::string* err);
    RuntimeOptions opts_;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    mtmd_context* mtmd_ctx_ = nullptr;
    bool ready_ = false;
    std::string last_error_;
    std::string device_;
    std::string precision_;
    int32_t threads_ = 0;
    bool cpu_mode_ = false;
    DecoderConfig cfg_;

    // Physical backends via the shared lease registry; the pixel-decoder
    // graphs reuse the resolved lease handle (session-local graphs/buffers).
    aicore::runtime::BackendLease gpu_lease_, cpu_lease_;

    // ---- pixel-decoder pass state (loaded lazily on first use) ----
    bool load_decoder_weights(std::string* err);
    bool run_pixel_pass(const RgbImage& rgb,
                        const std::string& prompt,
                        const std::string& response,
                        const std::vector<int32_t>& obj_positions,
                        const std::vector<ResultBox>& boxes,
                        RunResult* out,
                        std::string* err);
    // Decoder-graph pass shared by the full pixel path and the stages-in
    // parity harness: consumes host stages, builds+runs the decoder graph,
    // and fills the matte plus (when dump_dir is set) the parity dumps.
    // Decoder-parity harness: feed torch-dumped stages directly.
    bool run_stages_in(const std::string& dir,
                       const std::vector<ResultBox>& boxes,
                       RunResult* out,
                       std::string* err);

    bool run_decoder_graph(const RgbImage& rgb,
                           std::vector<float>&& cnn,
                           std::vector<std::vector<float>>&& level_maps,
                           std::vector<float>&& t_seg,
                           const std::vector<ResultBox>& boxes,
                           int32_t q,
                           int64_t S,
                           int64_t npx,
                           int64_t npy,
                           int64_t n_embd_vit,
                           int64_t n_embd,
                           RunResult* out,
                           std::string* err);

    struct VlmOutput {
        std::string text;
        std::vector<int32_t> ids;  // generated token ids
        int32_t n_prompt = 0;      // chat chunk count (incl. image tokens)
        int32_t n_chat = 0;        // chat token positions (for teacher force)
    };
    // rgb is consumed through the mtmd bitmap path (the supported VLM route).
    bool run_vlm(const RgbImage& rgb,
                 const std::string& prompt,
                 VlmOutput* out,
                 std::string* err);

    std::unique_ptr<GgufFile> decoder_gguf_;  // sidecar weights (host)
    std::unique_ptr<GgufFile> vision_gguf_;
    RgbImage rgb_for_stages_{};          // 1x1 stand-in for the stages-in matte
    std::vector<float> boxes01_direct_;  // stages-in: xyxy from the torch dump
    std::vector<float> query_position_direct_;  // stages-in: MSDA isolation
                                                // query position override
    ggml_context* wctx_ = nullptr;              // weight leaves (per run)
    ggml_backend_t pixel_backend_ = nullptr;
    ggml_backend_dev_t pixel_dev_ = nullptr;
    MaskDecoderWeights dw_{};
};

}  // namespace wwv
