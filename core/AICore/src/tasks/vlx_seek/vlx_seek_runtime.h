// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// VLX-Seek task runtime: owns the detector, the vision stack and the LLM and
// runs the full chain per request. Ported from the upstream
// Asher-1/VLX-Seek-GGML cpp_ggml/src/vlx_seek_runtime.* and adapted to the
// AICore conventions: device resolution and admission through the shared
// runtime layer, logging through aicore_log, no process exits, and the
// fused-CUDA RoIAlign hooks removed (the host RoIAlign path is the
// functionally identical path for every backend).
//
// The upstream CLI (vlx-cli) is intentionally not ported: its argument
// parsing/printing role is covered by the qVLXSeek batch command and plugin.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vlx_pipeline_common.h"  // ResultBox, BoxD
#include "vlx_types.h"

namespace vlx {

struct VlxSeekOptions {
    std::string device = "auto";  // AICore runtime order (CUDA->Vulkan->CPU)
    int threads = 0;              // 0 = backend default
    std::string precision = "f16";
    std::string models_dir;       // required: directory holding the GGUFs
    std::string wedetect_gguf;    // empty = derive from models_dir (f32 always)
    std::string vision_gguf;      // empty = vlx-vision-10b-<precision>.gguf
    std::string llm_gguf;         // empty = vlx-10b-llm-<precision|f16>.gguf
    int max_new_tokens = 2048;
    std::string dump_dir;         // empty = no parity dumps
    int log_level = 1;            // 0=DEBUG 1=INFO 2=WARN 3=ERROR (aicore gate)
};

// One inference request. rgb is tightly-packed RGB8 (the runtime's native
// format; the C API converts other aicore_image_view layouts first).
struct VlxRunRequest {
    const uint8_t* rgb = nullptr;
    int width = 0;
    int height = 0;
    std::string task = "detection";
    std::string text;
    std::string lang = "en";
    const float* boxes = nullptr;  // xyxy quadruples; NULL = run WeDetect
    int32_t n_boxes = 0;
    const int32_t* target_indexes = nullptr;  // region tasks only
    int32_t n_targets = 0;
    int warm_runs = 0;     // extra vision passes (steady-state benchmarking)
};

struct VlxRunResult {
    int code = 0;  // 0 ok, -1 contract/validation, -2 runtime, -3 resources
    std::string error;
    std::string answer;
    std::string json;  // official inference.py JSON
    std::vector<ResultBox> boxes;
    int32_t prompt_tokens = 0;
    int32_t completion_tokens = 0;
    int32_t proposal_count = 0;
    // per-stage timings (ms), aicore_pipeline_timings semantics
    double preprocess_ms = 0, inference_ms = 0, postprocess_ms = 0,
           serialization_ms = 0, e2e_ms = 0;
    // per-stage vision/llm split of inference_ms (informational)
    double vision_ms = 0, llm_ms = 0;
};

class VlxSeekRuntime {
public:
    explicit VlxSeekRuntime(const VlxSeekOptions& opts);
    ~VlxSeekRuntime();
    VlxSeekRuntime(const VlxSeekRuntime&) = delete;
    VlxSeekRuntime& operator=(const VlxSeekRuntime&) = delete;

    // Loads the detector, the vision stack and the LLM. Returns false with
    // last_error() set on failure (missing GGUF, device admission, ...).
    bool load();
    bool ready() const;
    const std::string& last_error() const;
    const std::string& device() const;
    int threads() const;
    const std::string& precision() const;

    // Full chain for one request. The typed result is also retained by the
    // runtime until the next call (info_json reads it).
    VlxRunResult run(const VlxRunRequest& req);

    // Official-style render of the retained result over the given RGB image
    // into a PNG (debug/dump path; the plugin renders through Qt instead).
    // Returns false on write failure.
    bool render_png(const VlxRunResult& result, const uint8_t* rgb, int width,
                    int height, const std::string& out_png_path);

    std::string info_json() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace vlx
