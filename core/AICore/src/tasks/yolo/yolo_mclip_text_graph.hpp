#pragma once

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// M-CLIP multilingual text encoder on ggml — sentence-transformers
// clip-ViT-B-32-multilingual-v1: a 6-layer DistilBERT tower whose mean-pooled
// hidden state is linearly projected into the OpenAI CLIP ViT-B/32 *text*
// space. Because the projection target is exactly the space YOLO-World's
// detection head was trained against, the head consumes these embeddings
// unchanged — prompts in 100+ languages work with zero detector changes.
//
// Interface mirrors clip::text_* (same 512-d L2-normalised output contract);
// the tower is selected per text-GGUF (the "mclip.arch" KV) in capi.cpp, so
// the English CLIP path is untouched.
//
// Deviations from the CLIP tower (inherent to DistilBERT, both handled):
//  * bidirectional (non-causal) attention with a post-LN residual layout;
//  * WordPiece tokenizer (119547-entry multilingual vocab) instead of the
//    CLIP SimpleTokenizer — Chinese is encoded per-character;
//  * per-string graph build at the true sequence length (no padding, hence
//    no attention mask and exact mean pooling).
//
// Encoding runs once per class list on a CPU-only private backend bundle
// (identical policy to the CLIP tower) — no steady-state VRAM cost and no
// effect on per-frame inference latency.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct gguf_context;

#include "tasks/yolo/backend.hpp"

namespace mclip {

constexpr int EMBED_DIM = 768;   // DistilBERT hidden size
constexpr int OUT_DIM = 512;     // projected CLIP ViT-B/32 text space
constexpr int MAX_TOKENS = 256;  // hard token cap incl. [CLS]/[SEP]
constexpr int N_LAYERS = 6;
constexpr int N_HEADS = 12;

// Text-encoder session: weights + WordPiece tables. The compute graph is
// built per encode call at the true sequence length (see cpp header note).
struct TextSession {
    ggml_context* wctx = nullptr;  // weight tensor structs (host data)
    yolo::BackendCtx backend;      // PRIVATE CPU-only bundle (never leased:
                                   // same isolation rule as the CLIP tower)

    ggml_tensor* word_emb = nullptr;  // f16 [V, D]
    ggml_tensor* pos_emb = nullptr;   // f16 [MAX_POS, D]
    ggml_tensor* emb_ln_w = nullptr;  // f32 [D]
    ggml_tensor* emb_ln_b = nullptr;  // f32 [D]
    ggml_tensor* proj_w = nullptr;    // f16/f16 [OUT_DIM, D]
    ggml_tensor* proj_b = nullptr;    // optional f32 [OUT_DIM] (fitted
                                      // affine bridges; absent = no bias)

    struct Block {
        ggml_tensor* attn_in_w;  // f16 [3*D, D] fused q|k|v
        ggml_tensor* attn_in_b;  // f32 [3*D]
        ggml_tensor* attn_out_w;
        ggml_tensor* attn_out_b;
        ggml_tensor* sa_ln_w;
        ggml_tensor* sa_ln_b;
        ggml_tensor* ffn_in_w;  // [4*D, D]
        ggml_tensor* ffn_in_b;
        ggml_tensor* ffn_out_w;  // [D, 4*D]
        ggml_tensor* ffn_out_b;
        ggml_tensor* out_ln_w;
        ggml_tensor* out_ln_b;
    };
    Block blocks[N_LAYERS];

    // WordPiece tables (loaded from the GGUF KV arrays).
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> encoder;
    int cls_id = 101;
    int sep_id = 102;
    int unk_id = 100;

    // Per-encode graph (built at the true sequence length; freed and rebuilt
    // when the length changes — see build_and_compute in the cpp).
    ggml_cgraph* text_graph = nullptr;
    ggml_tensor* text_output_embed = nullptr;  // f32 [OUT_DIM]
    ggml_context* text_gctx = nullptr;
};

// Create a text-encoder session from an mclip GGUF. Returns nullptr on error
// (reason logged).
TextSession* text_create_session(const std::string& gguf_path, int threads = 0);

// Free a session previously created with text_create_session. Safe on NULL.
void text_free_session(TextSession* s);

// WordPiece-tokenize `text` into at most `cap` ids ([CLS] ... [SEP]).
// Returns the number of tokens written.
int text_tokenize(TextSession* s, const char* text, int32_t* tokens, int cap);

// Encode `n_tokens` ids into a 512-d L2-normalised embedding (builds the
// graph at the exact sequence length — no padding).
bool text_encode_tokens(TextSession* s, const int32_t* tokens, int n_tokens,
                        float* embed);

// Convenience: text_tokenize + text_encode_tokens in one call.
bool text_encode_string(TextSession* s, const char* text, float* embed);

}  // namespace mclip
