#pragma once


// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: AGPL-3.0
// ----------------------------------------------------------------------------
//
// YOLO inference session. In-tree port of ultralytics-ggml
// cpp_ggml/src/yolo_graph.{hpp,cpp}.
//
// Two departures from the upstream single-shot session, both forced by the
// ACloudViewer usage pattern (GUI dialog + live video stream):
//
//  1. The upstream create_session() reloads the GGUF and rebuilds the whole
//     backend stack per image. Here the model definition (host weight
//     copies), the weight tensor structs (wctx), the weight buffer (wbuf)
//     and the BackendCtx live for the whole session; only the graph context
//     is rebuilt, and only when the letterbox canvas changes size. A
//     fixed-resolution video stream keeps one graph for its whole life.
//  2. The upstream YOLO_USE_VULKAN / YOLO_USE_CUDA compile-time dispatch is
//     gone: the graph is always built with the generic op vocabulary
//     (im2col + mul_mat for quantized convs, conv_2d_dw with F16 kernels,
//     F32 input) and the [gpu, cpu] scheduler routes each op at runtime.
//     Input is always F32; ggml cast ops and the scheduler handle dtype.

#include <string>
#include <vector>

#include "tasks/yolo/backend.hpp"

#include "tasks/yolo/yolo_common.hpp"

#include "tasks/yolo/yolo_gguf_loader.hpp"


struct ggml_backend_buffer;
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;

namespace yolo {

/* An inference session: parsed GGUF model + persistent weights + the compute
 * graph for the current letterbox canvas. */
struct Session {
    ModelDef model;  // parsed GGUF: ops + host weight copies (kept for rebuilds)

    BackendCtx backend;                  // leased backends (+ sched on GPU)
    ggml_context* wctx = nullptr;        // weight tensor structs (data in wbuf)
    ggml_backend_buffer_t wbuf = nullptr;

    SessionOptions opts;                 // creation-time configuration
    bool q8_direct = false;  // one-shot load decision (CUDA/Vulkan f16 flow);
                             // reused by every canvas rebuild

    /* Run plan for the current canvas — rebuilt by session_ensure_canvas().
     * All tensor structs and the cgraph live inside gctx. */
    ggml_context* gctx = nullptr;
    ggml_tensor* input = nullptr;   // external [W, H, 3] F32 letterboxed image
    ggml_tensor* output = nullptr;  // detect [A, no] or metric depth [W, H, 1, 1]
    ggml_tensor* output_proto = nullptr;  // segment protos [W, H, nm, 1]
    ggml_cgraph* graph = nullptr;
    int input_w = 0;                // current canvas dims (stride-multiple,
    int input_h = 0;                // non-square under LetterBox auto=True)
    std::vector<ggml_fp16_t> output_f16;        // F16 readback scratch
    std::vector<ggml_fp16_t> output_proto_f16;  // segment proto F16 scratch

    // Postprocess constants (mirrors ultralytics make_anchors). Rebuilt with
    // the graph because the anchor grid depends on the canvas dims.
    std::vector<float> anchors;        // [A*2] (x+0.5, y+0.5) per anchor
    std::vector<float> anchor_strides; // [A]
    int anchor_total = 0;
    std::vector<float> dfl_proj;       // [reg_max]

    // Gap-profiler state (only touched when opts.profile_gaps).
    double gap_up_ms = 0.0, gap_comp_ms = 0.0;
    double gap_get_ms = 0.0, gap_cast_ms = 0.0;
    int gap_frames = 0, gap_rframes = 0;
};

// Create a session for a GGUF model on the requested device
// ("auto" | "cpu" | "cuda" | "vulkan" | "metal" | ...). `threads` <= 0 means
// one thread. The graph is initially built for the square imgsz stored in
// the GGUF metadata. Returns nullptr on failure (reason logged).
Session* create_session(const std::string& gguf_path,
                        const std::string& device_request,
                        const SessionOptions& opts = {});

// Ensure the session's graph matches the letterbox canvas (input_w x
// input_h). Rebuilds the graph context when the canvas changed, reusing the
// session's weights and backend; a no-op otherwise. Returns false on failure
// (the session is then left without a usable plan; see create_session).
bool session_ensure_canvas(Session* s, int input_w, int input_h);

// Drop the host-side copies of the model weights. The device weight buffer
// (wbuf) is untouched, so inference and canvas rebuilds keep working; this
// only halves the session's host memory footprint. session_ensure_host_weights
// reloads the copies from the GGUF on demand.
bool session_release_host_weights(Session* s);

// Reload released host weight copies from the GGUF file (recorded offsets,
// no full-file mapping) and re-run the backend preprocessing. No-op when the
// copies are present. Returns false when the GGUF cannot be reopened.
bool session_ensure_host_weights(Session* s);

// Release a session. Safe to call on nullptr.
void free_session(Session* s);

// Copy a CHW float image into the input tensor, run the graph.
bool session_run(Session* s, const float* chw_image);

// Read back the raw detect output [no, A] (row-major: no rows x A anchors).
// For segment models `no` additionally carries nm mask-coefficient rows.
bool session_read_output(Session* s, std::vector<float>& out, int& no, int& na);

// Read back the segment mask prototypes [nm, H, W] (row-major, canvas/4 grid).
bool session_read_proto(Session* s, std::vector<float>& out, int& nm, int& w,
                        int& h);

// Read back a metric depth map in meters, row-major [height, width].
bool session_read_depth(Session* s, std::vector<float>& out, int& width,
                        int& height);

}  // namespace yolo
