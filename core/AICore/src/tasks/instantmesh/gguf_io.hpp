// Minimal GGUF reader used across model loaders.
//
// Models are stored as GGUF (see convert/). Every model writes a
// `general.architecture` tag plus its own hparams KV block; the loader reads
// metadata first, then allocates weights on the chosen backend and streams the
// payloads directly from file (the standard llama.cpp / stable-diffusion.cpp
// path: `no_alloc = true` then `ggml_backend_tensor_alloc`).
#ifndef INSTANTMESH_CORE_GGUF_IO_HPP
#define INSTANTMESH_CORE_GGUF_IO_HPP

#include <cstdint>
#include <string>
#include <unordered_map>

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

namespace instantmesh {

// RAII wrapper around a gguf_context + ggml_context.
struct GgufModel {
    gguf_context * gguf = nullptr;
    ggml_context * ctx  = nullptr;
    // backend buffer holding the streamed weights (freed in unload()).
    ggml_backend_buffer_t buffer = nullptr;
    // name -> tensor (built once at load for O(1) graph wiring later).
    std::unordered_map<std::string, ggml_tensor *> tensors;

    ~GgufModel();
    // Release the weight buffer + metadata early while the object stays alive
    // (tensor map is cleared; the model must not be used afterwards). Lets a
    // pipeline drop stages whose weights are no longer needed — e.g. the
    // zero123pp UNet/CLIP before the VAE decode — to cut the VRAM peak.
    void unload();
    // Move owned contiguous weights without re-reading files or changing dtype/layout.
    // The caller holds the involved backend leases and has no in-flight graph using these tensors.
    bool move_weights(ggml_backend_t destination, std::string * error);
};

// KV readers with defaults (return `def` if the key is absent).
uint32_t kv_u32(const gguf_context * g, const char * key, uint32_t def);
int32_t  kv_i32(const gguf_context * g, const char * key, int32_t def);
float    kv_f32(const gguf_context * g, const char * key, float def);
bool     kv_bool(const gguf_context * g, const char * key, bool def);
const char * kv_str(const gguf_context * g, const char * key, const char * def);

// Opens a GGUF, allocates all tensors on `backend`, and fills `tensors`.
// Returns false (and sets `error`) on failure.
bool load_gguf(const std::string & path, ggml_backend_t backend,
               GgufModel & out, std::string * error);

} // namespace instantmesh

#endif // INSTANTMESH_CORE_GGUF_IO_HPP