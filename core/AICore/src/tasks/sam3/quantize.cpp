// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// SAM3 / SAM2 GGUF weight quantization.  Ported from the upstream
// sam3-ggml examples/quantize.cpp whose decision rule mirrors the register_*
// macros: quantize 2D weights whose leading dimension is block-aligned and
// whose name is not an embedding / bias / norm parameter.

#include "tasks/sam3/quantize.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common/aicore_log.hpp"
#include "ggml.h"
#include "gguf.h"

namespace aicore {
namespace sam3 {
namespace {

// clang-format off
bool should_quantize(const std::string& name, ggml_type file_type) {
    // Tensors already quantized — skip.
    if (ggml_is_quantized(file_type)) return false;

    // Embedding tensors that MUST stay F32 (they are lookup tables, positional
    // encodings, special tokens, etc.). This list mirrors the upstream
    // example's name_contains filter and the T2f/T3f/T4f registers in sam3.cpp.
    auto em = [&](const char* sub) { return name.find(sub) != std::string::npos; };
    if (em("token_embed")   || em("pos_embed")
        || em("query_embed")   || em("label_embed")
        || em("cls_embed")     || em("point_embeddings")
        || em("not_a_point_embed") || em("no_mask_embed")
        || em("no_mem_embed")  || em("no_obj_embed")
        || em("presence_token.weight")
        || em("iou_token")     || em("mask_tokens")
        || em("obj_score_token")
        || em("pe_gaussian")   || em("freqs_cis")
        || em("gamma")         || em("tpos_enc")
        || em("no_obj_ptr")    || em("no_mem_pos_enc")
        || em("trk_mask_ds")   || em("latents")) {
        return false;
    }

    // 1D parameters (biases, layer-norm scale/shift) are always F32.
    if (em(".bias") || em("norm")) return false;

    return true;
}
// clang-format on

bool parse_type(const std::string& s, ggml_type& out) {
    std::string lc = s;
    for (auto& c : lc)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    if (lc == "q4_0") {
        out = GGML_TYPE_Q4_0;
        return true;
    }
    if (lc == "q4_1") {
        out = GGML_TYPE_Q4_1;
        return true;
    }
    if (lc == "q8_0") {
        out = GGML_TYPE_Q8_0;
        return true;
    }
    return false;
}

}  // namespace

bool quantize_gguf(const std::string& input_gguf,
                   const std::string& output_gguf,
                   const std::string& type_name) {
    ggml_type qtype = GGML_TYPE_F32;
    if (!parse_type(type_name, qtype)) {
        AICORE_LOG_ERROR(
                "[sam3] ",
                "quantize: unknown type '%s' (expected q4_0/q4_1/q8_0)\n",
                type_name.c_str());
        return false;
    }

    // ── Read source GGUF ────────────────────────────────────────────────
    ggml_context* meta_ctx = nullptr;
    gguf_init_params params{/*no_alloc=*/false, /*ctx=*/&meta_ctx};
    gguf_context* src = gguf_init_from_file(input_gguf.c_str(), params);
    if (!src || !meta_ctx) {
        AICORE_LOG_ERROR("[sam3] ", "quantize: failed to open '%s'\n",
                         input_gguf.c_str());
        if (src) gguf_free(src);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    const int64_t n_tensors = gguf_get_n_tensors(src);
    AICORE_LOG_PRINT("[sam3] ", "quantize: %lld tensors -> %s\n",
                     (long long)n_tensors, ggml_type_name(qtype));

    // ── Output GGUF: copy all metadata, set the weight-type KV ──────────
    gguf_context* out = gguf_init_empty();
    gguf_set_kv(out, src);
    gguf_set_val_i32(out, "sam3.ftype", (int32_t)qtype);

    const int blk_size = ggml_blck_size(qtype);

    // Scratch ctx for output tensor descriptors (not data; owners live in
    // the vector below).
    ggml_init_params ep{};
    ep.mem_size = ggml_tensor_overhead() * (size_t)(n_tensors + 8);
    ep.mem_buffer = nullptr;
    ep.no_alloc = true;
    ggml_context* out_ctx = ggml_init(ep);
    if (!out_ctx) {
        AICORE_LOG_ERROR("[sam3] ", "quantize: ggml_init for out_ctx failed\n");
        gguf_free(out);
        gguf_free(src);
        ggml_free(meta_ctx);
        return false;
    }

    ggml_quantize_init(qtype);

    // Owned byte buffers — must outlive gguf_write_to_file because
    // gguf_add_tensor stores a data pointer, not a copy.
    std::vector<std::vector<uint8_t>> owners;
    owners.reserve((size_t)n_tensors);

    std::vector<float> f32_buf;
    int n_quant = 0, n_kept = 0;
    bool failed = false;

    for (int64_t ti = 0; ti < n_tensors && !failed; ++ti) {
        const char* name = gguf_get_tensor_name(src, ti);
        ggml_tensor* src_t = ggml_get_tensor(meta_ctx, name);
        if (!src_t || !src_t->data) {
            AICORE_LOG_ERROR("[sam3] ", "quantize: tensor '%s' has no data\n",
                             name);
            failed = true;
            break;
        }

        const int64_t ne[GGML_MAX_DIMS] = {src_t->ne[0], src_t->ne[1],
                                           src_t->ne[2], src_t->ne[3]};
        // Row = ne[0]; the quantize decision needs row alignment.
        const int64_t nrows = ggml_nelements(src_t) / src_t->ne[0];
        ggml_type out_type = src_t->type;
        std::vector<uint8_t> bytes;

        const bool quant = should_quantize(name, src_t->type) &&
                           ggml_n_dims(src_t) == 2 && nrows > 0 &&
                           src_t->ne[0] % blk_size == 0;

        if (quant) {
            // Dequantize to F32
            f32_buf.clear();
            const int64_t n_el = ggml_nelements(src_t);
            f32_buf.resize(n_el);
            if (src_t->type == GGML_TYPE_F32) {
                std::memcpy(f32_buf.data(), src_t->data,
                            (size_t)n_el * sizeof(float));
            } else if (src_t->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row(
                        static_cast<const ggml_fp16_t*>(src_t->data),
                        f32_buf.data(), n_el);
            } else {
                const auto* tr = ggml_get_type_traits(src_t->type);
                if (!tr || !tr->to_float) {
                    AICORE_LOG_ERROR("[sam3] ",
                                     "quantize: cannot dequantize '%s' (%s)\n",
                                     name, ggml_type_name(src_t->type));
                    failed = true;
                    break;
                }
                tr->to_float(src_t->data, f32_buf.data(), n_el);
            }

            // Quantize
            out_type = qtype;
            const size_t qsz =
                    ggml_row_size(qtype, src_t->ne[0]) * (size_t)nrows;
            bytes.resize(qsz);
            const size_t got =
                    ggml_quantize_chunk(qtype, f32_buf.data(), bytes.data(), 0,
                                        nrows, src_t->ne[0], nullptr);
            if (got != qsz) {
                AICORE_LOG_ERROR("[sam3] ",
                                 "quantize: size mismatch '%s' (%zu vs %zu)\n",
                                 name, got, qsz);
                failed = true;
                break;
            }
            ++n_quant;
        } else {
            // Copy as-is
            const size_t nb = ggml_nbytes(src_t);
            bytes.assign(static_cast<const uint8_t*>(src_t->data),
                         static_cast<const uint8_t*>(src_t->data) + nb);
            ++n_kept;
        }

        ggml_tensor* dst =
                ggml_new_tensor(out_ctx, out_type, ggml_n_dims(src_t), ne);
        ggml_set_name(dst, name);
        owners.emplace_back(std::move(bytes));
        dst->data = owners.back().data();
        gguf_add_tensor(out, dst);
    }

    bool ok = !failed;
    if (ok) {
        if (!gguf_write_to_file(out, output_gguf.c_str(),
                                /*only_meta=*/false)) {
            AICORE_LOG_ERROR("[sam3] ",
                             "quantize: gguf_write_to_file failed for '%s'\n",
                             output_gguf.c_str());
            ok = false;
        }
    }

    if (ok) {
        AICORE_LOG_PRINT("[sam3] ", "quantize: %d quantized (%s), %d kept\n",
                         n_quant, ggml_type_name(qtype), n_kept);
    }

    ggml_free(out_ctx);
    gguf_free(out);
    gguf_free(src);
    ggml_free(meta_ctx);
    return ok;
}

}  // namespace sam3
}  // namespace aicore