#ifndef YOLO_COMMON_HPP
#define YOLO_COMMON_HPP

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

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ggml.h"

namespace yolo {

// Logging -------------------------------------------------------------------

enum class LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

void logf(LogLevel level, const char* fmt, ...);
extern int g_log_level;

#define YOLO_LOG_DEBUG(...) ::yolo::logf(::yolo::LogLevel::DEBUG, __VA_ARGS__)
#define YOLO_LOG_INFO(...) ::yolo::logf(::yolo::LogLevel::INFO, __VA_ARGS__)
#define YOLO_LOG_WARN(...) ::yolo::logf(::yolo::LogLevel::WARN, __VA_ARGS__)
#define YOLO_LOG_ERROR(...) ::yolo::logf(::yolo::LogLevel::ERROR, __VA_ARGS__)

// Detection result -----------------------------------------------------------

struct Detection {
    float x1, y1, x2, y2;  // pixels in the original input image
    float score;           // max class probability after sigmoid
    int class_id;
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
    std::string task;  // "detect" | "depth"
    std::string dtype;
    int nc = 80;
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
// the GGUF file.
struct HostTensor {
    std::vector<uint8_t> data;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};  // ggml order: ne[0] fastest
    std::string name;
};

struct ModelDef {
    ModelMeta meta;
    std::vector<OpDef> ops;
    std::map<std::string, HostTensor> tensors;

    // Flattened per-level head info, taken from the single detect op.
    bool has_detect = false;
    int detect_op_index = -1;
};

}  // namespace yolo

#endif
