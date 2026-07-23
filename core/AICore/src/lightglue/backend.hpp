// Persistent ggml backend + graph allocator for LightGlue matching.
#pragma once

#include <ggml-backend.h>

#include <string>

namespace aicore {
namespace lightglue {

struct engine_backend {
    ggml_backend_t be = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::string device;
    std::string error;

    bool init(const std::string& device_req, int n_threads);
    void release();
    ~engine_backend() { release(); }

    bool is_cpu() const;
};

}  // namespace lightglue
}  // namespace aicore
