#pragma once


// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: AGPL-3.0
// ----------------------------------------------------------------------------
//
// YOLO task shared data structures (in-tree port of ultralytics-ggml
// cpp_ggml: https://github.com/Asher-1/ultralytics-ggml).
//
// The upstream source is AGPL-3.0; this port keeps the license until a
// written relicensing decision is recorded (see
// ultralytics-ggml-integration-plan.md §5).

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common/aicore_log.hpp"

#include "ggml.h"

namespace yolo {

// Logging -------------------------------------------------------------------
// Routes through aicore_log.hpp, which forwards to CVLog when the AICore
// target is built with AICore_HAS_CVLOG (ACloudViewer Console) and falls
// back to stderr otherwise. Runtime level filtering lives in the shared
// aicore_log_at() gate (AICORE_LOG_LEVEL_*); there is no task-private level
// enum or thread-local state.

void logf(int level, const char* fmt, ...);
void set_log_level(int level);
int get_log_level();

#define YOLO_LOG_DEBUG(...)                                                    \
    ::yolo::logf(AICORE_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define YOLO_LOG_INFO(...)                                                     \
    ::yolo::logf(AICORE_LOG_LEVEL_INFO, __VA_ARGS__)
#define YOLO_LOG_WARN(...)                                                     \
    ::yolo::logf(AICORE_LOG_LEVEL_WARN, __VA_ARGS__)
#define YOLO_LOG_ERROR(...)                                                    \
    ::yolo::logf(AICORE_LOG_LEVEL_ERROR, __VA_ARGS__)

// SessionOptions: explicit typed configuration (no env vars) -----------------

struct SessionOptions {
    int threads = 0;             // <=0: hardware default
    int input_w = 0;             // 0: square imgsz stored in the GGUF metadata
    int input_h = 0;
    int log_level = 1;           // 0=DEBUG,1=INFO,2=WARN,3=ERROR
    bool keep_all_ops = false;   // debug: keep every op output alive
    bool profile_ops = false;    // debug: per-op wall-time table
    bool profile_gaps = false;   // debug: per-stage timing on stderr
};

// Detection result -----------------------------------------------------------

struct Detection {
    float x1, y1, x2, y2;  // pixels in the original input image
    float score;           // max class probability after sigmoid
    int class_id;
    int anchor = -1;       // index into the raw [no, na] output; -1 = not tracked
};

// Letterbox geometry shared by detect (box unscaling) and depth (canvas
// restore): one description must serve both tasks for a given input.
struct LetterboxInfo {
    float scale;   // resized / original
    int pad_w;     // left padding in pixels (right may differ by 1px)
    int pad_h;
    int new_w;     // resized dims before padding
    int new_h;
    int imgsz_w;   // padded canvas dims
    int imgsz_h;
};

// Model metadata from GGUF ----------------------------------------------------

struct ModelMeta {
    std::string name;
    std::string task;  // "detect" | "segment" | "depth"
    std::string dtype;
    int nc = 80;
    int nm = 0;          // mask prototypes (0 for detect/depth)
    int nl = 3;
    int imgsz = 640;
    int reg_max = 16;
    bool end2end = false;
    int max_det = 300;
    std::vector<float> strides;
    std::vector<std::string> class_names;
};

// GGUF op-graph vocabulary ----------------------------------------------------

// One node of the op-graph vocabulary written by the upstream converter
// (scripts/convert_yolo_to_gguf.py).
struct OpDef {
    std::string type;  // conv|dwconv|maxpool|concat|upsample|interpolate|conv_transpose|add|slice|psa_attention|detect|depth
    std::vector<int> inputs;                     // op indices; -1 = graph input image
    std::map<std::string, int64_t> iparams;      // ints
    std::map<std::string, double> fparams;       // floats
    std::map<std::string, std::vector<int64_t>> aparams;  // int arrays (s/p/d)
    std::map<std::string, std::string> sparams;  // strings (act)
    std::vector<std::string> tensor_names;       // w, b, qkv_w, ...

    int64_t ip(const std::string& k, int64_t def = 0) const {
        auto it = iparams.find(k);
        return it == iparams.end() ? def : it->second;
    }
    int64_t ai(const std::string& k, int idx, int64_t def = 0) const {
        auto it = aparams.find(k);
        if (it == aparams.end() || idx >= (int)it->second.size()) return def;
        return it->second[idx];
    }
};

// Host-side weight: ggml_type + raw block data + logical shape (torch order).
// The host copy is kept for the context lifetime so the graph can be rebuilt
// when the letterbox canvas (graph input shape) changes without re-reading
// the GGUF file. Once the device weight buffer exists, rebuilds reuse that
// buffer and never touch the host copy, so callers may drop the host copy
// via session_release_host_weights / aicore_yolo_release_host_weights to
// halve the memory footprint; session_ensure_host_weights reloads it from
// the GGUF on demand.
struct HostTensor {
    std::vector<uint8_t> data;
    ggml_type type = GGML_TYPE_F32;   // current (possibly backend-preprocessed)
    ggml_type file_type = GGML_TYPE_F32;  // original type stored in the GGUF
    int64_t ne[4] = {1, 1, 1, 1};     // ggml order: ne[0] fastest
    std::string name;
    size_t file_offset = 0;  // absolute GGUF file offset (for on-demand reload)
};

struct ModelDef {
    ModelMeta meta;
    std::vector<OpDef> ops;
    std::map<std::string, HostTensor> tensors;
    std::string gguf_path;  // source file (for on-demand host weight reload)

    // Flattened per-level head info, taken from the single detect op.
    bool has_detect = false;
    int detect_op_index = -1;
};

// Timing helper ---------------------------------------------------------------

struct Clock {
    std::chrono::steady_clock::time_point t0;
    Clock() : t0(std::chrono::steady_clock::now()) {}
    double ms_since() const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0)
                .count();
    }
    static std::chrono::steady_clock::time_point now() {
        return std::chrono::steady_clock::now();
    }
};

inline double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t)
            .count();
}

}  // namespace yolo
