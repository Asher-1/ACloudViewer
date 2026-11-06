// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Task-internal encoder facade for token-consuming models (querysplat).
// AICore-private: no public ABI, no Qt — the same in-tree delegation pattern
// as trellis -> rmbg. One facade owns the encoder GGUF, the physical backend
// lease, and one per-geometry VGGTRuntime with token export enabled. The
// consumer (querysplat task) reuses backend() for its own decoder graph so
// encoder and decoder share one device and one lease serialization scope.
//
// Lifecycle: load() -> ensure_geometry() -> run(); the runtime is rebuilt
// (weights re-staged) whenever (S, H, W) changes — the map_anything capi
// contract, where callers sweeping resolutions expect that cost.
#pragma once

#include "tasks/map_anything/map_anything_backend.hpp"

#include "tasks/map_anything/map_anything_graph.hpp"

#include "tasks/map_anything/map_anything_gguf_loader.hpp"


#include <memory>
#include <string>

namespace map_anything {

class EncoderRuntime {
   public:
    struct Config {
        std::string encoder_gguf;  // vggt-omega family GGUF path
        std::string device = "auto";
        int n_threads = 0;         // CPU backend thread count
        // torch-parity profile (see RuntimeOptions::f32_accuracy). The
        // querysplat chain defaults to it; explicit per options.
        bool f32_accuracy = true;
        // Debug-only stage-dump passthrough (MAP_ANYTHING_ENABLE_DUMP builds).
        std::string dump_dir;
    };

    EncoderRuntime() = default;
    ~EncoderRuntime();  // runtime -> model -> backend, in that order

    EncoderRuntime(const EncoderRuntime&) = delete;
    EncoderRuntime& operator=(const EncoderRuntime&) = delete;

    // Loads the GGUF and initializes the backend. Returns false and fills
    // *error (when non-null) on any failure; the object stays reusable via
    // a fresh load() call.
    bool load(const Config& cfg, std::string* error);

    // (Re)builds the graph runtime when the (S, H, W) geometry changes.
    bool ensure_geometry(int n_views, int height, int width,
                         std::string* error);

    // One encoder pass. The caller must hold the backend lease lock (the
    // querysplat session holds it across encoder + decoder + rasterizer).
    bool run(const float* images_f32, VGGTOutputs& out, std::string* error);

    bool valid() const { return model_ != nullptr && be_.valid(); }
    bool ready_for(int n_views, int height, int width) const;
    const Backend& backend() const { return be_; }
    int n_threads() const { return be_.n_threads; }
    const RuntimeOptions& runtime_options() const { return ropts_; }

   private:
    void teardown_runtime();
    void teardown_all();

    Config cfg_;
    Backend be_;
    std::unique_ptr<GGUFModel> model_;
    std::unique_ptr<VGGTRuntime> rt_;
    RuntimeOptions ropts_;
    int cur_s_ = 0, cur_h_ = 0, cur_w_ = 0;
    std::string error_;
};

}  // namespace map_anything
