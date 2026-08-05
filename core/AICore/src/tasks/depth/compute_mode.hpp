#pragma once

namespace aicore {
namespace depth {
// Global compute-mode flag, set by Engine::load once the backend's offload
// status is known. When true the graph builders route to STANDARD ggml ops
// that have CUDA kernels (direct conv2d, manual attention) instead of the
// CPU-tuned custom paths (Winograd custom-op, F32 flash-attn), which keeps the
// GPU graph free of forced GPU->CPU round-trips. Defaults to false, so the CPU
// path is byte-identical to before this flag existed.
void set_gpu_mode(bool on);
bool gpu_mode();
} // namespace depth
} // namespace aicore
