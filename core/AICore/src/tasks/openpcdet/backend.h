// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// AICore adaptation of OpenPCDet-GGML's engine backend bundle
// (https://github.com/Asher-1/OpenPCDet-GGML, cpp_ggml/src/backend.h).
//
// The upstream engine built its own ggml backend instances; here the physical
// backend handles are shared process-wide through the AICore runtime lease
// registry (aicore::runtime::BackendLease + ggml_common), matching the
// rfdetr/yolo task pattern. The public surface (init/release/is_cpu/...)
// is kept identical so the ported model code stays byte-comparable with
// upstream.
//
// One backend + one persistent gallocr live for the model's lifetime; each
// forward builds its graph in a no_alloc=true context, allocates via the
// persistent gallocr (compute buffer stays alive across calls), writes input
// data AFTER alloc, computes, reads back.
#pragma once

#include <ggml-backend.h>

#include <string>

#include "common/ggml_backend_registry.hpp"

namespace openpcdet {

struct engine_backend {
    engine_backend() = default;
    engine_backend(const engine_backend &) = delete;
    engine_backend &operator=(const engine_backend &) = delete;
    engine_backend(engine_backend &&) = delete;
    engine_backend &operator=(engine_backend &&) = delete;

    ggml_backend_t be = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::string device;  // resolved name
    std::string error;

    // device_req: "" | "cpu" | "gpu" | "cuda" | "vulkan" (optionally ":N").
    // "auto"/"gpu" follows the AICore platform order (CUDA -> Vulkan -> CPU on
    // Linux/Windows, Metal -> CPU on macOS). Returns false on failure (see
    // `error`).
    bool init(const std::string &device_req, int n_threads);
    void release();
    ~engine_backend() { release(); }

    bool is_cpu() const;
    bool is_cuda() const;
    bool is_vulkan() const;

private:
    aicore::runtime::BackendLease cpu_lease_;
    aicore::runtime::BackendLease gpu_lease_;
};

}  // namespace openpcdet
