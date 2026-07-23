// Persistent ggml backend + graph allocator for FreeSplatter.
// Adapted from free-splatter.cpp/src/backend.h to use ggml_common utilities.
#pragma once

#include <ggml-backend.h>

#include <cstddef>
#include <string>

namespace aicore {
namespace gaussian {

struct engine_backend {
    ggml_backend_t be = nullptr;
    ggml_backend_t cpu_be = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_backend_sched_t sched = nullptr;
    bool use_sched = false;
    std::string    device;   // resolved name
    std::string    error;

    // device_req accepts auto, cpu, gpu, sycl, vulkan, cuda, or metal;
    // GPU backends optionally take ":N" to select the Nth matching device.
    bool init(const std::string & device_req, int n_threads);
    bool alloc_graph(ggml_cgraph* graph, size_t graph_size);
    enum ggml_status compute_graph(ggml_cgraph* graph);
    void release();
    ~engine_backend() { release(); }

    bool is_cpu() const;
};

} // namespace gaussian
} // namespace aicore
