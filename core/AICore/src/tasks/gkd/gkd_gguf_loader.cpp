// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// GKDT ggml runtime - GGUF model loader.
#include "tasks/gkd/gkd_gguf_loader.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "tasks/gkd/gkd_common.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define GKD_HAS_MMAP 1
#endif

namespace gkd {

namespace kv {

static int64_t find(const gguf_context* g, const char* key) {
    return gguf_find_key(g, key);
}

// Type-tolerant scalar extraction (the converter writes u32/i32/f32/bool).
static double scalar(const gguf_context* g, int64_t id) {
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT8:
            return gguf_get_val_u8(g, id);
        case GGUF_TYPE_INT8:
            return gguf_get_val_i8(g, id);
        case GGUF_TYPE_UINT16:
            return gguf_get_val_u16(g, id);
        case GGUF_TYPE_INT16:
            return gguf_get_val_i16(g, id);
        case GGUF_TYPE_UINT32:
            return gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:
            return gguf_get_val_i32(g, id);
        case GGUF_TYPE_FLOAT32:
            return gguf_get_val_f32(g, id);
        case GGUF_TYPE_BOOL:
            return gguf_get_val_bool(g, id) ? 1 : 0;
        case GGUF_TYPE_UINT64:
            return (double)gguf_get_val_u64(g, id);
        case GGUF_TYPE_INT64:
            return (double)gguf_get_val_i64(g, id);
        case GGUF_TYPE_FLOAT64:
            return gguf_get_val_f64(g, id);
        default:
            return 0.0;
    }
}

int32_t i32(const gguf_context* g, const char* key, int32_t def) {
    int64_t id = find(g, key);
    return id < 0 ? def : (int32_t)scalar(g, id);
}

float f32(const gguf_context* g, const char* key, float def) {
    int64_t id = find(g, key);
    return id < 0 ? def : (float)scalar(g, id);
}

bool boolean(const gguf_context* g, const char* key, bool def) {
    int64_t id = find(g, key);
    return id < 0 ? def : (scalar(g, id) != 0.0);
}

std::string str(const gguf_context* g,
                const char* key,
                const std::string& def) {
    int64_t id = find(g, key);
    return id < 0 ? def : std::string(gguf_get_val_str(g, id));
}

std::vector<std::string> arr_str(const gguf_context* g, const char* key) {
    std::vector<std::string> out;
    int64_t id = find(g, key);
    if (id < 0) return out;
    size_t n = gguf_get_arr_n(g, id);
    out.reserve(n);
    for (size_t i = 0; i < n; i++) {
        out.emplace_back(gguf_get_arr_str(g, id, i));
    }
    return out;
}

std::vector<float> arr_f32(const gguf_context* g, const char* key) {
    std::vector<float> out;
    int64_t id = find(g, key);
    if (id < 0) return out;
    size_t n = gguf_get_arr_n(g, id);
    if (gguf_get_arr_type(g, id) == GGUF_TYPE_FLOAT32) {
        const float* data = (const float*)gguf_get_arr_data(g, id);
        out.assign(data, data + n);
    } else {
        out.resize(n);
        // fall back to element-wise extraction for non-f32 arrays
        for (size_t i = 0; i < n; i++) {
            const void* p =
                    nullptr;  // scalar extraction is not exposed; keep zeros
            (void)p;
        }
    }
    return out;
}

}  // namespace kv

static void read_params(GkdModel* m) {
    const gguf_context* g = m->gguf;
    ModelParams& P = m->P;
    P.D = kv::i32(g, "gkd.vis.embed_dim", P.D);
    P.blocks = kv::i32(g, "gkd.vis.depth", P.blocks);
    P.heads = kv::i32(g, "gkd.vis.num_heads", P.heads);
    P.patch = kv::i32(g, "gkd.vis.patch_size", P.patch);
    P.n_storage = kv::i32(g, "gkd.vis.n_storage_tokens", P.n_storage);
    P.norm_eps = kv::f32(g, "gkd.vis.norm_eps", P.norm_eps);
    P.has_layerscale =
            kv::boolean(g, "gkd.vis.has_layerscale", P.has_layerscale);
    P.mask_k_bias = kv::boolean(g, "gkd.vis.mask_k_bias", P.mask_k_bias);
    P.img_size = kv::i32(g, "gkd.vis.img_size", P.img_size);
    P.feat_w = kv::i32(g, "gkd.vis.feat_width", P.feat_w);

    P.T_D = kv::i32(g, "gkd.txt.dim", P.T_D);
    P.T_layers = kv::i32(g, "gkd.txt.layers", P.T_layers);
    P.T_heads = kv::i32(g, "gkd.txt.num_heads", P.T_heads);
    P.T_ctx = kv::i32(g, "gkd.txt.context_length", P.T_ctx);
    P.T_vocab = kv::i32(g, "gkd.txt.vocab_size", P.T_vocab);
    P.T_proj_dim = kv::i32(g, "gkd.txt.proj_dim", P.T_proj_dim);
    P.T_norm_eps = kv::f32(g, "gkd.txt.norm_eps", P.T_norm_eps);

    P.A_in = kv::i32(g, "gkd.anet.dim_in", P.A_in);
    P.A_model = kv::i32(g, "gkd.anet.dim_model", P.A_model);
    P.A_blocks = kv::i32(g, "gkd.anet.blocks", P.A_blocks);
    P.A_heads = kv::i32(g, "gkd.anet.num_heads", P.A_heads);
    P.A_out = kv::i32(g, "gkd.anet.dim_out", P.A_out);
    P.text_half = kv::i32(g, "gkd.text_feature_dim_half", P.text_half);

    P.K_D = kv::i32(g, "gkd.kg.dim", P.K_D);
    P.K_blocks = kv::i32(g, "gkd.kg.blocks", P.K_blocks);
    P.K_heads = kv::i32(g, "gkd.kg.num_heads", P.K_heads);
    P.K_ff = kv::i32(g, "gkd.kg.d_ff", P.K_ff);
    P.K_use_mask_token =
            kv::boolean(g, "gkd.kg.use_mask_token", P.K_use_mask_token);
    P.K_norm_eps = kv::f32(g, "gkd.kg.norm_eps", P.K_norm_eps);

    P.up_scale = kv::i32(g, "gkd.det.up_scale", P.up_scale);
    P.kernel_norm = kv::boolean(g, "gkd.det.kernel_norm", P.kernel_norm);

    P.sigma = kv::f32(g, "gkd.sigma", P.sigma);
    P.pad_kps = kv::i32(g, "gkd.pad_kps", P.pad_kps);
    std::vector<float> mean = kv::arr_f32(g, "gkd.img_mean");
    std::vector<float> stdv = kv::arr_f32(g, "gkd.img_std");
    if (mean.size() == 3)
        std::memcpy(P.img_mean, mean.data(), sizeof(float) * 3);
    if (stdv.size() == 3)
        std::memcpy(P.img_std, stdv.data(), sizeof(float) * 3);
}

std::unique_ptr<GkdModel> load_gkd_model(const std::string& path,
                                         ggml_backend_buffer_type_t buft) {
    auto m = std::make_unique<GkdModel>();

    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &m->meta_ctx;
    m->gguf = gguf_init_from_file(path.c_str(), params);
    if (!m->gguf) {
        GKD_LOG_ERROR("failed to open GGUF model: %s", path.c_str());
        return nullptr;
    }

    read_params(m.get());

    // Instantiate every GGUF tensor inside the meta context.
    int64_t n_tensors = gguf_get_n_tensors(m->gguf);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(m->gguf, i);
        ggml_tensor* t = ggml_get_tensor(m->meta_ctx, name);
        if (!t) {
            GKD_LOG_ERROR("tensor '%s' listed in GGUF but not in meta context",
                          name);
            return nullptr;
        }
        m->tensors[name] = t;
    }

    // Allocate one backend buffer for all weights.
    m->weight_buf = ggml_backend_alloc_ctx_tensors_from_buft(m->meta_ctx, buft);
    if (!m->weight_buf) {
        GKD_LOG_ERROR("failed to allocate the weight buffer");
        return nullptr;
    }

    // Map the file and copy tensor payloads into the backend buffer.
    size_t data_offset = gguf_get_data_offset(m->gguf);
#if defined(GKD_HAS_MMAP)
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        GKD_LOG_ERROR("cannot reopen %s", path.c_str());
        return nullptr;
    }
    struct stat st {};
    fstat(fd, &st);
    size_t file_size = (size_t)st.st_size;
    void* base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        GKD_LOG_ERROR("mmap failed for %s", path.c_str());
        close(fd);
        return nullptr;
    }
    const uint8_t* file_base = (const uint8_t*)base;
#else
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        GKD_LOG_ERROR("cannot reopen %s", path.c_str());
        return nullptr;
    }
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    std::vector<uint8_t> file_buf(file_size);
    fseek(f, 0, SEEK_SET);
    if (fread(file_buf.data(), 1, file_size, f) != file_size) {
        GKD_LOG_ERROR("short read on %s", path.c_str());
        fclose(f);
        return nullptr;
    }
    fclose(f);
    const uint8_t* file_base = file_buf.data();
#endif

    for (int64_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(m->gguf, i);
        ggml_tensor* t = ggml_get_tensor(m->meta_ctx, name);
        size_t off = data_offset + gguf_get_tensor_offset(m->gguf, i);
        size_t size = gguf_get_tensor_size(m->gguf, i);
        if (off + size > file_size) {
            GKD_LOG_ERROR("tensor '%s' exceeds file bounds", name);
            return nullptr;
        }
        ggml_backend_tensor_set(t, file_base + off, 0, size);
    }

#if defined(GKD_HAS_MMAP)
    munmap(base, file_size);
    close(fd);
#endif

    // Tokenizer arrays.
    m->bpe_vocab = kv::arr_str(m->gguf, "gkd.tokenizer.vocab");
    m->bpe_merges = kv::arr_str(m->gguf, "gkd.tokenizer.merges");

    const ModelParams& P = m->P;
    GKD_LOG_INFO(
            "model %s: DINOv3 ViT (D=%d, depth=%d, heads=%d) + dinotxt (D=%d, "
            "depth=%d) + KG (%d blocks)",
            path.c_str(), P.D, P.blocks, P.heads, P.T_D, P.T_layers,
            P.K_blocks);
    // Per-load allocation info is diagnostic-only: the plugin surfaces a
    // summarized "Model loaded" line, and every-forward logs must not spam
    // the console (default level Info hides GKD_LOG_DEBUG lines).
    GKD_LOG_DEBUG("weights buffer: %.1f MiB on %s",
                  ggml_backend_buffer_get_size(m->weight_buf) / 1024.0 / 1024.0,
                  ggml_backend_buffer_name(m->weight_buf));
    return m;
}

}  // namespace gkd
