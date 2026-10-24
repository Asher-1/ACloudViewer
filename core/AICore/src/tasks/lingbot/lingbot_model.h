// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend;
typedef struct ggml_backend *ggml_backend_t;

namespace lingbot {

struct model_info {
    int image_size = 0;
    int patch_size = 0;
    int embed_dim = 0;
    int block_count = 0;
    int output_dim = 0;
    std::string weight_type;
    std::string graph_version;
};

struct output {
    std::vector<float> pose_enc;
    std::vector<float> depth;
    std::vector<float> depth_conf;
    // Official pose_encoding_to_extri_intri equivalent: C2W 4x4 matrices
    // and [fx, fy, cx, cy], one contiguous item per output frame.
    std::vector<float> c2w;
    std::vector<float> intrinsics;
    int batch = 0;
    int frames = 0;
    int height = 0;
    int width = 0;
};

// Persistent-cache attention mode.
//   none   : exact F32 cache, hand-written F32 attention (bit-level route)
//   strict : F16 cache upload, hand-written F32 attention (parity route)
//   flash  : F16 cache, ggml_flash_attn_ext fast path (CUDA/Turing parity,
//            see benchmarks/validation_report.md "Flash-Attention Parity
//            Route")
enum class kv_f16_mode { none, strict, flash };

// Explicit inference configuration. Every field replaces a former
// LINGBOT_* environment switch (the env reads were removed; the CLI maps
// command-line flags onto these fields, see tools/lingbot-map-cli.cpp).
struct model_options {
    // ---- cache profile & mode (affects inference values) ----
    int kv_cache_scale = 1;   // persistent scale frames
    int kv_cache_window = 4;  // sliding-window frames
    kv_f16_mode kv_f16 = kv_f16_mode::none;
    bool kv_resident = true;   // device-resident cache (F16 modes)
    int kv_total_frames = 0;   // 0 -> infer() frames (capacity hint)
    int num_scale_frames = 0;  // 0 -> kv_cache_scale
    // Official keyframe policy (demo.py --keyframe_interval): every N-th
    // streaming frame persists its KV in the cache; non-keyframes attend to
    // [cache | own KV] and discard. Scale frames are always cached. The
    // official demo auto-selects ceil(N/320) for streaming runs above 320
    // frames.
    int keyframe_interval = 1;
    bool disable_kv_cache = false;
    bool force_f32_weights = false;  // dequant every weight to F32 buffers
    bool use_dpt_pos = true;         // DPT positional embedding
    // ---- backend behavior (affects speed / backend selection) ----
    bool vulkan_fast = false;         // non-parity coopmat/f16 opt-out
    bool vulkan_integer_dot = false;  // exact-integer dp4a q8 GEMMs
    bool repeat_run = false;          // CLI benchmark repeat loop
    // ---- diagnostics (logging / dumps only; no effect on values) ----
    bool debug_kv = false;
    bool debug_infer = false;
    bool profile = false;
    bool flat_off = false;
    bool dump_internal = false;
    bool dump_stages = false;
    bool dump_camera_stages = false;
    bool dump_features = false;
    std::string dump_block;  // restrict dump_internal to a block index
    int dump_at = -1;        // restrict dumps to one inference index
    std::string dump_features_dir;
    std::string dump_stages_dir;
    std::string dump_camera_stages_dir;
};

class model {
public:
    model();
    ~model();
    model(model &&) noexcept;
    model &operator=(model &&) noexcept;
    model(const model &) = delete;
    model &operator=(const model &) = delete;

    bool load(const std::string &gguf_path,
              const std::string &backend,
              int threads,
              const model_options &options = {});
    bool infer(const float *images,
               int batch,
               int frames,
               int channels,
               int height,
               int width,
               output *result);
    // Per-frame streaming callback. Invoked exactly once per completed output
    // frame while infer() runs (scale-pass frames invoke it once per frame in
    // that pass, streaming frames once each). `frame` is the zero-based
    // output-frame index; `frame_out` carries that frame's pose_enc (9),
    // depth (H*W), depth_conf (H*W), c2w (16) and intrinsics (4) slices.
    // Called on the inference thread before the next frame starts. Returning
    // false aborts inference (infer() then returns false with error
    // "frame callback aborted").
    void set_frame_callback(
            std::function<bool(int frame, const output &frame_out)> cb);
    void reset_cache();
    const model_info &info() const;
    // Backend-RESOLVED device name ("CUDA0", "Vulkan0", "cpu", ...); set by
    // load() from the AICore lease (AICore adaptation; empty before load).
    const std::string &device_name() const;
    // The physical backend this model executes on (owned by the AICore
    // lease; shared with companion models such as the native skyseg).
    ggml_backend_t backend_handle() const;
    // Effective CPU thread count passed to the backend lease.
    int threads() const;
    const std::string &error() const;
    void release();

private:
    struct impl;
    std::unique_ptr<impl> p_;
};

}  // namespace lingbot
