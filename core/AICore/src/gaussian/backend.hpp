// Persistent ggml backend + graph allocator for FreeSplatter.
// Adapted from free-splatter.cpp/src/backend.h to use ggml_common utilities.
#pragma once

#include <ggml-backend.h>

#include <string>

namespace aicore {
namespace gaussian {

struct engine_backend {
    ggml_backend_t be     = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::string    device;   // resolved name
    std::string    error;

    // device_req: "" | "cpu" | "gpu" | "cuda" | "vulkan", optionally ":N" to pick
    // the Nth matching GPU. "gpu" = first GPU of whichever backend is built.
    bool init(const std::string & device_req, int n_threads);
    void release();
    ~engine_backend() { release(); }

    bool is_cpu() const;
};

} // namespace gaussian
} // namespace aicore
