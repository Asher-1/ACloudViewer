#pragma once

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// AICore adaptation of RMBG-2.0-GGML's public model wrapper. The upstream
// single-owner ggml_backend_t is replaced by a process-shared
// aicore::runtime::BackendLease so device selection follows the AICore
// runtime ("auto" = CUDA -> Vulkan -> CPU on Linux/Windows, Metal -> CPU on
// macOS) and the physical backend is shared with the other AICore tasks.

#include <string>
#include <vector>

#include "ggml-backend.h"
#include "common/ggml_backend_registry.hpp"

#include "tasks/rmbg/rmbg_graph.hpp"


namespace rmbg {

struct Config {
    int input_size = 1024;
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};
    std::string backbone = "swin_v1_l";
};

class RmbgDeviceGraph;

struct Model {
    Model() = default;
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;

    Config cfg;
    aicore::runtime::BackendLease lease;  // process-shared ggml backend
    ggml_backend_t backend = nullptr;     // lease.handle() alias
    RmbgDeviceGraph *graph = nullptr;
    std::string backend_name;
    std::string math_profile;
    int n_threads = 0;
    bool graph_ready = false;
};

// Normalize a math profile name to one of
// "default" | "optimized" | "strict" | "fast" | "unsafe-fast"
// (unknown/empty -> "optimized", the historical default).
std::string normalize_math_profile(const char *profile);

// Load a unified RMBG-2.0 GGUF. `math_profile` selects the Vulkan/CUDA math
// configuration (replaces the RMBG_* environment variables); the implied
// ggml-side environment overrides are queued through the AICore env bridge
// before the backend registers. `graph_options` carries the fine-tuning
// switches; the profile always wins for the flow-defining fields.
bool load_gguf(const char *path,
               const char *device,
               int n_threads,
               const char *math_profile,
               const GraphOptions &graph_options,
               Model &out,
               std::string &err);
void free_model(Model &m);
bool remove_background(Model &m,
                       const void *image_bytes,
                       int image_len,
                       std::vector<uint8_t> &out_png,
                       std::string &err);

}  // namespace rmbg
