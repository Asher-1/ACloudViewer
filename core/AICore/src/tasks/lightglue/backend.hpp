// Persistent ggml backend + graph allocator for LightGlue matching.
#pragma once

#include <ggml-backend.h>

#include <string>

#include "ggml_backend_registry.hpp"

namespace aicore {
namespace lightglue {

struct engine_backend {
    aicore::runtime::BackendLease lease;
    // Non-owning shorthand retained to keep graph code session-local.
    ggml_backend_t be = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::string device;
    std::string error;

    bool init(const std::string& device_req, int n_threads);
    void release();
    ~engine_backend() { release(); }

    bool is_cpu() const;
    bool supports_fused_attention() const;
    std::unique_lock<std::recursive_mutex> lock() const { return lease.lock(); }
};

}  // namespace lightglue
}  // namespace aicore
