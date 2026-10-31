// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "gguf_io.hpp"

#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace instantmesh {

GgufModel::~GgufModel() { unload(); }

void GgufModel::unload() {
    if (buffer) ggml_backend_buffer_free(buffer);
    buffer = nullptr;
    if (gguf) gguf_free(gguf);
    gguf = nullptr;
    if (ctx) ggml_free(ctx);
    ctx = nullptr;
    tensors.clear();
}

bool GgufModel::move_weights(ggml_backend_t destination, std::string* error) {
    const auto fail = [&](const char* message) {
        if (error) *error = message;
        return false;
    };
    if (!destination || !ctx || !buffer || tensors.empty())
        return fail("Invalid owned weight relocation");
    const auto type = ggml_backend_get_default_buffer_type(destination);
    if (ggml_backend_buffer_get_type(buffer) == type) return true;
    // Build a complete destination transaction while the original tensors
    // remain valid. No file read, numeric conversion, backend cache reset, or
    // foreign model mutation occurs.
    ggml_init_params params{ggml_tensor_overhead() * (tensors.size() + 1),
                            nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> next_ctx(
            ggml_init(params), ggml_free);
    if (!next_ctx) return fail("Weight relocation metadata allocation failed");
    std::unordered_map<std::string, ggml_tensor*> next_tensors;
    for (const auto& entry : tensors) {
        if (!ggml_is_contiguous(entry.second) || entry.second->view_src)
            return fail("Weight relocation requires owned contiguous tensors");
        auto* next = ggml_dup_tensor(next_ctx.get(), entry.second);
        ggml_set_name(next, entry.first.c_str());
        next_tensors.emplace(entry.first, next);
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>
            next_buffer(
                    ggml_backend_alloc_ctx_tensors(next_ctx.get(), destination),
                    ggml_backend_buffer_free);
    if (!next_buffer) return fail("Weight relocation device allocation failed");
    ggml_backend_buffer_set_usage(next_buffer.get(),
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto& entry : tensors)
        ggml_backend_tensor_copy(entry.second, next_tensors.at(entry.first));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    buffer = next_buffer.release();
    ctx = next_ctx.release();
    tensors.swap(next_tensors);
    return true;
}

uint32_t kv_u32(const gguf_context* g, const char* key, uint32_t def) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32:
            return (uint32_t)gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:
            return (uint32_t)gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64:
            return (uint32_t)gguf_get_val_u64(g, id);
        case GGUF_TYPE_INT64:
            return (uint32_t)gguf_get_val_i64(g, id);
        case GGUF_TYPE_FLOAT32:
            return (uint32_t)gguf_get_val_f32(g, id);
        default:
            return def;
    }
}
int32_t kv_i32(const gguf_context* g, const char* key, int32_t def) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32:
            return (int32_t)gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:
            return (int32_t)gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64:
            return (int32_t)gguf_get_val_u64(g, id);
        case GGUF_TYPE_INT64:
            return (int32_t)gguf_get_val_i64(g, id);
        case GGUF_TYPE_FLOAT32:
            return (int32_t)gguf_get_val_f32(g, id);
        default:
            return def;
    }
}
float kv_f32(const gguf_context* g, const char* key, float def) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_FLOAT32:
            return gguf_get_val_f32(g, id);
        case GGUF_TYPE_FLOAT64:
            return (float)gguf_get_val_f64(g, id);
        case GGUF_TYPE_INT32:
            return (float)gguf_get_val_i32(g, id);
        case GGUF_TYPE_INT64:
            return (float)gguf_get_val_i64(g, id);
        case GGUF_TYPE_BOOL:
            return gguf_get_val_bool(g, id) ? 1.0f : 0.0f;
        default:
            return def;
    }
}
bool kv_bool(const gguf_context* g, const char* key, bool def) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_BOOL:
            return gguf_get_val_bool(g, id);
        case GGUF_TYPE_INT32:
            return gguf_get_val_i32(g, id) != 0;
        case GGUF_TYPE_INT64:
            return gguf_get_val_i64(g, id) != 0;
        default:
            return def;
    }
}
const char* kv_str(const gguf_context* g, const char* key, const char* def) {
    const int64_t id = gguf_find_key(g, key);
    return id < 0 ? def : gguf_get_val_str(g, id);
}

bool load_gguf(const std::string& path,
               ggml_backend_t backend,
               GgufModel& out,
               std::string* error) {
    // Parse metadata only; weights are streamed from file after allocation.
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = &out.ctx;

    out.gguf = gguf_init_from_file(path.c_str(), params);
    if (!out.gguf) {
        if (error) *error = std::string("gguf_init_from_file failed: ") + path;
        return false;
    }

    // Index tensors by name for O(1) graph wiring later.
    const int n_tensors = gguf_get_n_tensors(out.gguf);
    for (int i = 0; i < n_tensors; ++i) {
        ggml_tensor* t =
                ggml_get_tensor(out.ctx, gguf_get_tensor_name(out.gguf, i));
        if (t) out.tensors.emplace(ggml_get_name(t), t);
    }

    // Allocate every tensor on the chosen backend (single buffer).
    out.buffer = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (out.buffer == nullptr) {
        if (error) *error = "ggml_backend_alloc_ctx_tensors failed: " + path;
        return false;
    }

    // Stream weight payloads from the file into the backend memory.
    // std::ifstream (not fseek/long): zero123pp UNet f32 exceeds 2 GiB and
    // long offsets overflow on Windows.
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (error) *error = "fopen failed: " + path;
        return false;
    }
    std::vector<uint8_t> buf;
    for (int i = 0; i < n_tensors; ++i) {
        ggml_tensor* t = out.tensors[gguf_get_tensor_name(out.gguf, i)];
        const std::streamoff offs =
                (std::streamoff)(gguf_get_data_offset(out.gguf) +
                                 gguf_get_tensor_offset(out.gguf, i));
        const size_t nbytes = ggml_nbytes(t);
        buf.resize(nbytes);
        f.seekg(offs, std::ios::beg);
        if (!f.read(reinterpret_cast<char*>(buf.data()),
                    (std::streamsize)nbytes)) {
            if (error)
                *error = "read failed for tensor: " +
                         std::string(ggml_get_name(t));
            return false;
        }
        ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
    }
    return true;
}

}  // namespace instantmesh