// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Base detector framework. A `model` owns a model_file (weights + hparams),
// an engine_backend (ggml backend + gallocr), and a forward() that builds the
// ggml graph, allocates, computes, and reads results back. The per-architecture
// forward lives in models/<name>.cpp; this header defines the shared
// scaffolding.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "backend.h"
#include "gguf_loader.h"

namespace openpcdet {

// A detection result for one scene. Matches OpenPCDet's pred_dict layout.
struct detection {
    float x, y, z;     // box center
    float dx, dy, dz;  // box size
    float heading;     // box yaw
    float score;
    int32_t label;
};

struct scene_input {
    const float* points = nullptr;
    int32_t n_points = 0;
    int32_t point_dim = 0;
    const float* images = nullptr;
    int32_t num_cameras = 0;
    int32_t image_height = 0;
    int32_t image_width = 0;
    int32_t image_channels = 0;
    const float* lidar_to_image = nullptr;
    const float* camera_to_lidar = nullptr;
    const float* camera_intrinsics = nullptr;
    const float* image_augmentation = nullptr;
    const float* lidar_augmentation = nullptr;
};

// Sink for a named intermediate, given as row-major [n_rows, n_cols] f32.
// Called once per tap during forward and freed immediately, so a full-depth
// tapped run never holds all activations in host memory at once.
using tap_sink = std::function<void(const std::string& name,
                                    const float* data,
                                    int64_t n_rows,
                                    int64_t n_cols)>;

struct model {
    model_file file;
    engine_backend be;

    ggml_backend_buffer_t weights_buf = nullptr;
    ggml_context* device_ctx = nullptr;

    std::string error;

    virtual bool load(const std::string& gguf_path,
                      const std::string& device,
                      int n_threads);
    void release();
    virtual ~model() { release(); }

    hparams& hp() { return file.hp; }
    const hparams& hp() const { return file.hp; }

    // Run the full forward pass. `points` is a flat [N, point_dim] float32
    // array (point_dim == num_point_features, typically 4 = x,y,z,intensity).
    // Returns detected boxes in `out_dets` (already NMS'd). If `sink` is set,
    // every named tap is streamed to it (slower: marks them as graph outputs).
    virtual bool forward(const float* points,
                         int32_t n_points,
                         int32_t point_dim,
                         std::vector<detection>& out_dets,
                         const tap_sink& sink = {}) = 0;

    // Image-capable detectors override this method. The default keeps every
    // existing LiDAR-only graph source-compatible with ABI v2.
    virtual bool forward_scene(const scene_input& scene,
                               std::vector<detection>& out_dets,
                               const tap_sink& sink = {});

protected:
    bool realize_weights();
    // After realize_weights(), the file.ctx tensor pointers are remapped to the
    // device mirror (non-CPU). Subclasses capture their weight pointers in
    // map_tensors() BEFORE realize_weights() — so we provide a helper to
    // re-resolve a tensor by name on the active context.
    ggml_tensor* resolved(const char* name) const;
};

}  // namespace openpcdet
