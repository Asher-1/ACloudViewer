// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "gguf_loader.h"

namespace aicore {
namespace gaussian {

static const char* ARCH = "free-splatter";

namespace {

// KV readers: set err once on first failure, no-op afterwards (lets the caller
// batch reads and check a single error string).
struct kv {
    const gguf_context* g;
    std::string& err;

    int64_t find(const char* key, bool required) const {
        int64_t id = gguf_find_key(g, key);
        if (id < 0 && required && err.empty()) {
            err = std::string("missing GGUF key: ") + key;
        }
        return id;
    }
    void u32(const char* key, int32_t& out, bool required = true) const {
        int64_t id = find(key, required);
        if (id >= 0) out = (int32_t)gguf_get_val_u32(g, id);
    }
    void f32(const char* key, float& out, bool required = true) const {
        int64_t id = find(key, required);
        if (id >= 0) out = gguf_get_val_f32(g, id);
    }
    void boolean(const char* key, bool& out, bool required = true) const {
        int64_t id = find(key, required);
        if (id >= 0) out = gguf_get_val_bool(g, id);
    }
    std::string str(const char* key, bool required = true) const {
        int64_t id = find(key, required);
        if (id < 0) {
            return "";
        }
        const char* v = gguf_get_val_str(g, id);
        return v ? std::string(v) : std::string();
    }
};

std::string akey(const char* suffix) {
    return std::string(ARCH) + "." + suffix;
}

}  // namespace

bool model_file::open(const std::string& path, bool with_data) {
    close();

    gguf_init_params params = {/*no_alloc =*/!with_data, /*ctx =*/&ctx};
    guf = gguf_init_from_file(path.c_str(), params);
    if (!guf) {
        error = "failed to read GGUF: " + path;
        return false;
    }

    kv r{guf, error};

    const std::string arch = r.str("general.architecture");
    if (error.empty() && arch != ARCH) {
        error = "unsupported architecture '" + arch + "' (want " + ARCH + ")";
    }

    r.u32(akey("block_count").c_str(), hp.n_layer);
    r.u32(akey("embedding_length").c_str(), hp.n_embd);
    r.u32(akey("feed_forward_length").c_str(), hp.n_ff);
    r.u32(akey("attention.head_count").c_str(), hp.n_head);
    r.u32(akey("attention.key_length").c_str(), hp.head_dim);
    r.u32(akey("vision.patch_size").c_str(), hp.patch_size);
    r.u32(akey("vision.image_size").c_str(), hp.image_size);
    r.u32(akey("vision.in_channels").c_str(), hp.in_channels);
    r.u32(akey("vision.gaussian_channels").c_str(), hp.gaussian_channels);
    r.u32(akey("sh_degree").c_str(), hp.sh_degree);
    r.boolean(akey("sh_residual").c_str(), hp.sh_residual);
    r.boolean(akey("use_2dgs").c_str(), hp.use_2dgs);
    r.f32(akey("attention.layer_norm_epsilon").c_str(), hp.ln_eps);
    r.f32(akey("scale_min_act").c_str(), hp.scale_min_act);
    r.f32(akey("scale_max_act").c_str(), hp.scale_max_act);

    if (error.empty()) {
        if (hp.patch_size <= 0 || hp.image_size <= 0 ||
            hp.image_size % hp.patch_size != 0) {
            error = "invalid patch/image size (image_size must be a positive "
                    "multiple of patch_size)";
        } else if (hp.n_head <= 0 || hp.head_dim <= 0 ||
                   hp.n_head * hp.head_dim != hp.n_embd) {
            error = "head_count * key_length must equal embedding_length";
        }
    }

    if (!error.empty()) {
        close();
        return false;
    }
    return true;
}

void model_file::close() {
    if (guf) {
        gguf_free(guf);
        guf = nullptr;
    }
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    hp = hparams{};
    error.clear();
}

ggml_tensor* model_file::tensor(const char* name) const {
    return ctx ? ggml_get_tensor(ctx, name) : nullptr;
}

ggml_tensor* model_file::require(const char* name) {
    ggml_tensor* t = tensor(name);
    if (!t && error.empty()) {
        error = std::string("missing tensor: ") + name;
    }
    return t;
}

}  // namespace gaussian
}  // namespace aicore
