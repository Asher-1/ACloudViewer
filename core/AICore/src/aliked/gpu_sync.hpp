#pragma once

#include "../backend.h"

#include <ggml.h>

#include <string>

namespace lightglue::aliked_internal {

// Enable parity-gated Vulkan COMPUTE/SDDH/POST defaults (idempotent; opt-out via =0).
// Call on backend.Init("vulkan") and GpuPipelineCache::Warmup.
void ApplyVulkanAlikedPerfDefaults();

// No-op on Vulkan when defer sync is enabled (custom ops sync on entry).
void SyncGpuPipeline(internal::Backend *backend);

// Always drain the backend queue (pipeline section boundaries).
void FlushGpuPipeline(internal::Backend *backend);

// Vulkan: hard flush; CUDA/other: deferred sync when enabled.
void BarrierGpuPipeline(internal::Backend *backend);

// Re-bind gallocr to `graph` then compute. Pass `graph_gallocr` for cached graphs
// (each cache entry owns its gallocr); omit to use the backend's shared allocator.
bool GallocrComputeGraph(internal::Backend *backend, ggml_cgraph *graph,
                         std::string *error,
                         ggml_gallocr_t graph_gallocr = nullptr);

// Run a graph whose gallocr was bound once at build time (cached conv / op graphs).
// Vulkan re-binds gallocr on each call — required for safe multi-run reuse.
bool RunCachedGraphCompute(internal::Backend *backend, ggml_cgraph *graph,
                           std::string *error,
                           ggml_gallocr_t graph_gallocr = nullptr);

}  // namespace lightglue::aliked_internal
