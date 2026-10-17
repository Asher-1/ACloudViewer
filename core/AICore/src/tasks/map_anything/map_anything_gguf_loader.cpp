// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "map_anything_gguf_loader.hpp"

#include <cstring>

#include "gguf.h"

namespace map_anything {

namespace {

int get_i32(const gguf_context* g, const char* key, int def) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32:
            return (int)gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:
            return gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64:
            return (int)gguf_get_val_u64(g, id);
        case GGUF_TYPE_INT64:
            return (int)gguf_get_val_i64(g, id);
        default:
            return def;
    }
}

bool get_bool(const gguf_context* g, const char* key, bool def) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_BOOL:
            return gguf_get_val_bool(g, id);
        case GGUF_TYPE_UINT32:
            return gguf_get_val_u32(g, id) != 0;
        case GGUF_TYPE_INT32:
            return gguf_get_val_i32(g, id) != 0;
        default:
            return def;
    }
}

std::string get_str(const gguf_context* g, const char* key, const char* def) {
    const int64_t id = gguf_find_key(g, key);
    return id >= 0 ? std::string(gguf_get_val_str(g, id)) : std::string(def);
}

void parse_meta(const gguf_context* g, VGGTMeta& m) {
    // Normalize the architecture key FIRST, then dispatch to a
    // self-contained per-architecture branch: each branch only reads its own
    // key space, so a new architecture cannot silently inherit another one's
    // defaults. Unknown keys surface at builder lookup with a clear error
    // listing the registered architectures.
    m.architecture = get_str(g, "general.architecture", "vggt_omega");
    if (m.architecture == "vggt") {
        // Historical ambiguity: old vggt-omega files wrote
        // general.architecture="vggt", which is now the ORIGINAL VGGT's
        // key. Disambiguate by a feature tensor name: the omega dense head
        // has dp.interlayer.*, the original VGGT DPT has dp.scratch.* and a
        // separate point head (pt.*).
        bool has_point = false;
        const int64_t n_t = gguf_get_n_tensors(g);
        for (int64_t t = 0; t < n_t && !has_point; t++) {
            std::string tn = gguf_get_tensor_name(g, t);
            if (tn.rfind("pt.", 0) == 0) has_point = true;
        }
        m.architecture = has_point ? "vggt" : "vggt_omega";
    }

    m.name = get_str(g, "general.name", m.architecture.c_str());

    auto read_img_norm = [&](const char* mean_key, const char* std_key) {
        if (int64_t id = gguf_find_key(g, mean_key); id >= 0) {
            const size_t n = gguf_get_arr_n(g, id);
            const float* p = (const float*)gguf_get_arr_data(g, id);
            for (int i = 0; i < 3 && (size_t)i < n; i++) m.img_mean[i] = p[i];
        }
        if (int64_t id = gguf_find_key(g, std_key); id >= 0) {
            const size_t n = gguf_get_arr_n(g, id);
            const float* p = (const float*)gguf_get_arr_data(g, id);
            for (int i = 0; i < 3 && (size_t)i < n; i++) m.img_std[i] = p[i];
        }
    };

    if (m.architecture == "pi3" || m.architecture == "pi3x") {
        // Pi3 / Pi3X (yyfz233): same shape as the vggt family, own key
        // space. The decoder register count is 5 while num_register_tokens
        // keeps the ENCODER's 4 DINOv2 registers (the shared init math uses
        // Ncr = 1+4, which equals the decoder's register count by
        // coincidence of the two architectures).
        const std::string p = m.architecture;
        m.dtype = get_str(g, (p + ".dtype").c_str(), "?");
        m.patch_size = get_i32(g, (p + ".patch_size").c_str(), 14);
        m.embed_dim = get_i32(g, (p + ".embed_dim").c_str(), 1024);
        m.depth = get_i32(g, (p + ".enc_depth").c_str(), 24);
        m.aa_depth = get_i32(g, (p + ".dec_depth").c_str(), 36);
        m.num_heads = get_i32(g, (p + ".num_heads").c_str(), 16);
        m.num_register_tokens = 4;  // encoder DINOv2 registers
        m.head_depth = get_i32(g, (p + ".head_depth").c_str(), 5);
        read_img_norm((p + ".img_mean").c_str(), (p + ".img_std").c_str());
    } else if (m.architecture == "mapanything") {
        // MapAnything (facebook): dim 1536, heads 24, AAT inter-frame
        // blocks (is.*). num_register_tokens stays 0 (Ncr = 1 = the CLS
        // per-view token); head_depth is unused. The DPT consumes enc
        // final + IFR[0] + IFR[1] + final, i.e. cached_layer_idx[0]/[1]
        // only — [2]/[3] keep their inert default placeholders.
        m.dtype = get_str(g, "mapanything.dtype", "?");
        m.patch_size = get_i32(g, "mapanything.patch_size", 14);
        m.embed_dim = get_i32(g, "mapanything.embed_dim", 1536);
        m.depth = get_i32(g, "mapanything.enc_depth", 24);
        m.aa_depth = get_i32(g, "mapanything.is_depth", 16);
        m.num_heads = get_i32(g, "mapanything.num_heads", 24);
        m.num_register_tokens = 0;  // no encoder registers; Ncr = 1 = cls
        if (int64_t id = gguf_find_key(g, "mapanything.ifr_index0"); id >= 0)
            m.cached_layer_idx[0] = gguf_get_val_u32(g, id);
        if (int64_t id = gguf_find_key(g, "mapanything.ifr_index1"); id >= 0)
            m.cached_layer_idx[1] = gguf_get_val_u32(g, id);
        read_img_norm("mapanything.img_mean", "mapanything.img_std");
    } else if (m.architecture == "dust3r") {
        // DUSt3R (naver, M5 pair-wise third branch): embed_dim/num_heads/
        // depth keep the ENCODER's (CroCo ViT-L/16, RoPE100); the decoder
        // runs its own dims (BaseDecoder 768/12) via the dec_* fields
        // (defaults 0 — Dust3rImpl::build_graph hard-fails on a GGUF that
        // lacks them instead of silently running 768/12).
        m.dtype = get_str(g, "dust3r.dtype", "?");
        m.patch_size = get_i32(g, "dust3r.patch_size", 16);
        m.embed_dim = get_i32(g, "dust3r.enc_embed_dim", 1024);
        m.depth = get_i32(g, "dust3r.enc_depth", 24);
        m.num_heads = get_i32(g, "dust3r.enc_num_heads", 16);
        m.aa_depth = get_i32(g, "dust3r.dec_depth", 12);
        m.dec_embed_dim = get_i32(g, "dust3r.dec_embed_dim", 0);
        m.dec_num_heads = get_i32(g, "dust3r.dec_num_heads", 0);
        m.num_register_tokens = 0;
        read_img_norm("dust3r.img_mean", "dust3r.img_std");
    } else {
        // vggt-omega / original-VGGT (vggt.* key space; both keys were
        // written by convert_vggt_omega_to_gguf.py / convert_vggt_to_gguf.py
        // and the two builders share every vggt.* field except the point
        // head, which is detected by tensor presence, not metadata).
        m.dtype = get_str(g, "vggt.dtype", "?");
        m.patch_size = get_i32(g, "vggt.patch_size", 16);
        m.embed_dim = get_i32(g, "vggt.embed_dim", 1024);
        m.depth = get_i32(g, "vggt.depth", 24);
        m.aa_depth = get_i32(g, "vggt.aa_depth", 24);
        m.num_heads = get_i32(g, "vggt.num_heads", 16);
        m.num_register_tokens = get_i32(g, "vggt.num_register_tokens", 16);
        m.trunk_depth = get_i32(g, "vggt.trunk_depth", 4);
        m.dpt_features = get_i32(g, "vggt.dpt_features", 256);
        // omega text-alignment variant: the converter stores the flag next
        // to the text_alignment_head.* weights it keeps as f32/f16 extras
        m.enable_text_alignment =
                get_bool(g, "vggt.enable_text_alignment", false);

        if (int64_t id = gguf_find_key(g, "vggt.cached_layer_idx"); id >= 0) {
            const size_t n = gguf_get_arr_n(g, id);
            const int32_t* p = (const int32_t*)gguf_get_arr_data(g, id);
            for (size_t i = 0; i < n && i < 4; i++)
                m.cached_layer_idx[i] = p[i];
        }
        if (int64_t id = gguf_find_key(g, "vggt.register_attn_block_idx");
            id >= 0) {
            const size_t n = gguf_get_arr_n(g, id);
            const int32_t* p = (const int32_t*)gguf_get_arr_data(g, id);
            for (size_t i = 0; i < n && i < 5; i++)
                m.register_attn_block_idx[i] = p[i];
        }
        read_img_norm("vggt.img_mean", "vggt.img_std");
    }
}

}  // namespace

std::unique_ptr<GGUFModel> load_gguf(const std::string& path) {
    ggml_context* weight_ctx = nullptr;
    gguf_init_params ip{};
    ip.no_alloc = false;  // map tensor data directly
    ip.ctx = &weight_ctx;

    gguf_context* g = gguf_init_from_file(path.c_str(), ip);
    if (!g) {
        log_error("failed to open GGUF: %s", path.c_str());
        return nullptr;
    }

    auto model = std::make_unique<GGUFModel>();
    parse_meta(g, model->meta);

    const int64_t n_tensors = gguf_get_n_tensors(g);
    for (int64_t t = 0; t < n_tensors; t++) {
        const char* name = gguf_get_tensor_name(g, t);
        ggml_tensor* cur = ggml_get_tensor(weight_ctx, name);
        if (!cur) {
            log_error("tensor %s missing from ggml context", name);
            gguf_free(g);
            ggml_free(weight_ctx);
            return nullptr;
        }
        HostTensor ht;
        ht.name = name;
        ht.type = cur->type;
        for (int d = 0; d < 4; d++) ht.ne[d] = cur->ne[d];
        ht.data.resize(ggml_nbytes(cur));
        memcpy(ht.data.data(), cur->data, ggml_nbytes(cur));
        model->tensors[name] = std::move(ht);
    }

    log_info("loaded %s: %lld tensors, dtype=%s, patch=%d, dim=%d, depth=%d+%d",
             path.c_str(), (long long)n_tensors, model->meta.dtype.c_str(),
             model->meta.patch_size, model->meta.embed_dim, model->meta.depth,
             model->meta.aa_depth);
    // everything a normalization/metadata misconfiguration investigation
    // needs at first sight: the arch actually dispatched, the input norm
    // the graph will apply, and the dust3r-only decoder dims (0 = unused)
    log_info(
            "  arch=%s mean={%.3f,%.3f,%.3f} std={%.3f,%.3f,%.3f} "
            "dec_dim=%d dec_heads=%d",
            model->meta.architecture.c_str(), model->meta.img_mean[0],
            model->meta.img_mean[1], model->meta.img_mean[2],
            model->meta.img_std[0], model->meta.img_std[1],
            model->meta.img_std[2], model->meta.dec_embed_dim,
            model->meta.dec_num_heads);

    gguf_free(g);
    ggml_free(weight_ctx);
    return model;
}

}  // namespace map_anything
