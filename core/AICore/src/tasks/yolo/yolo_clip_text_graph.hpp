#pragma once


// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: AGPL-3.0
// ----------------------------------------------------------------------------
//
// CLIP (ViT-B/32) TEXT encoder inference on ggml — in-tree port of
// ultralytics-ggml cpp_ggml/src/clip_graph.{hpp,cpp}, reduced to the text
// path qYOLO needs (open-vocabulary class-name encoding for YOLO-World).
//
// Deviations from upstream (both deliberate):
//  * The visual encoder (image tower + clip_preprocess_image) is NOT ported:
//    the named use case is text conditioning only, and image preprocessing
//    lives on the plugin side in this repo (Qt QImage, see yolo_image.hpp).
//  * The upstream fprintf logging is replaced by the shared YOLO_LOG_*
//    macros (AICore logging rules); the backend context uses the in-tree
//    lease registry (CPU-only session, same as upstream's "no second GPU
//    init inside the YOLO process" constraint — solved process-wide by the
//    lease registry here).
//
// Usage:
//   clip::TextSession* s = clip::text_create_session("clip-ViT-B-32-f16.gguf");
//   float embed[512];
//   clip::text_encode_string(s, "person", embed);  // BPE + encode, end to end
//   clip::text_free_session(s);
//
// Precision: all internal computation runs in F32 for numerical stability;
// weight matrices are stored F16/Q8_0 in the GGUF and cast to F32 at graph
// build time (identical to the PyTorch CLIP text forward pass).

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct gguf_context;

#include "tasks/yolo/backend.hpp"

namespace clip {

// CLIP ViT-B/32 constants (mirror scripts/convert_clip_to_gguf.py).
constexpr int EMBED_DIM = 512;
constexpr int TEXT_CTX = 77;
constexpr int VOCAB_SIZE = 49408;
constexpr int TEXT_N_LAYERS = 12;
constexpr int TEXT_N_HEADS = 8;

// BPE tokenizer tables rebuilt from the GGUF KV arrays; shared by CLIP and
// MobileCLIP (both embed the same CLIP SimpleTokenizer tables).
struct ClipBpe {
    std::vector<std::string> vocab;    // tokens by id
    std::vector<std::pair<std::string, std::string>> merges;  // by rank
    std::unordered_map<std::string, int> encoder;  // token -> id
    std::unordered_map<std::string, std::string> bpe_cache;  // token -> merged
};

// Text-encoder session: weights + one causal transformer graph.
struct TextSession {
    ggml_context* wctx = nullptr;  // weight tensor structs (host data)
    yolo::BackendCtx backend;      // PRIVATE CPU-only bundle (never leased:
                                   // the text tower must stay fully isolated
                                   // from the process-wide lease registry,
                                   // matching the upstream "no shared backend
                                   // state inside the YOLO process" rule)
    bool owns_cpu_backend = false; // backend.cpu created here (freed here)

    ggml_tensor* text_embed_table = nullptr;  // token_embedding [49408, 512]
    ggml_tensor* text_pos_embed = nullptr;    // positional_embedding [77, 512]
    ggml_tensor* text_ln_final_w = nullptr;   // [512]
    ggml_tensor* text_ln_final_b = nullptr;   // [512]
    ggml_tensor* text_proj = nullptr;         // text_projection [512, 512]

    struct TextBlock {
        ggml_tensor* attn_in_w;   // [1536, 512] (qkv combined)
        ggml_tensor* attn_in_b;   // [1536]
        ggml_tensor* attn_out_w;  // [512, 512]
        ggml_tensor* attn_out_b;  // [512]
        ggml_tensor* mlp_fc_w;    // [2048, 512]
        ggml_tensor* mlp_fc_b;    // [2048]
        ggml_tensor* mlp_proj_w;  // [512, 2048]
        ggml_tensor* mlp_proj_b;  // [512]
        ggml_tensor* ln1_w;       // [512]
        ggml_tensor* ln1_b;       // [512]
        ggml_tensor* ln2_w;       // [512]
        ggml_tensor* ln2_b;       // [512]
    };
    TextBlock text_blocks[TEXT_N_LAYERS];

    // BPE tokenizer tables (loaded from the GGUF).
    ClipBpe bpe;

    // Text encoder graph.
    ggml_cgraph* text_graph = nullptr;
    ggml_tensor* text_input_tokens = nullptr;  // int32 [TEXT_CTX]
    ggml_tensor* text_eot_idx = nullptr;       // int32 [1] EOT position
    ggml_tensor* text_output_embed = nullptr;  // float [EMBED_DIM]
    ggml_context* text_gctx = nullptr;         // graph context
};

// Create a text-encoder session from a CLIP GGUF file. Returns nullptr on
// error (reason logged).
TextSession* text_create_session(const std::string& gguf_path, int threads = 0);

// Free a session previously created with text_create_session. Safe on NULL.
void text_free_session(TextSession* s);

// Encode tokenized text (TEXT_CTX int32 token ids, zero-padded) into a
// 512-d L2-normalised embedding.
bool text_encode_tokens(TextSession* s, const int32_t* tokens, float* embed);

// Tokenize a single text string into TEXT_CTX token ids exactly like the
// Python reference (clip.tokenize): lowercase -> BPE -> [sot, ..., eot] with
// truncation at TEXT_CTX. Returns the number of tokens written (<= TEXT_CTX).
int text_tokenize(TextSession* s, const char* text, int32_t* tokens);

// Convenience: text_tokenize + text_encode_tokens in one call.
bool text_encode_string(TextSession* s, const char* text, float* embed);

// ---------------------------------------------------------------------------
// Shared encoder building blocks (also used by the MobileCLIP tower)
// ---------------------------------------------------------------------------

// LayerNorm over ne0 with fixed eps=1e-5.
ggml_tensor* clip_layer_norm(ggml_context* ctx, ggml_tensor* x,
                             ggml_tensor* weight, ggml_tensor* bias);

// Multi-head self-attention; x is [D, S], causal enables the lower-
// triangular mask. Weights may be F16/Q8_0 (cast to F32 internally).
ggml_tensor* clip_self_attention(ggml_context* ctx, ggml_tensor* x,
                                 ggml_tensor* in_proj_w, ggml_tensor* in_proj_b,
                                 ggml_tensor* out_proj_w, ggml_tensor* out_proj_b,
                                 int n_heads, int d_head, bool causal);

// Two-layer MLP; exact_gelu selects erf-GELU (MobileCLIP) over QuickGELU
// (CLIP).
ggml_tensor* clip_mlp_block(ggml_context* ctx, ggml_tensor* x,
                            ggml_tensor* fc_w, ggml_tensor* fc_b,
                            ggml_tensor* proj_w, ggml_tensor* proj_b,
                            bool exact_gelu);

// L2-normalise the last dimension.
ggml_tensor* clip_l2_norm(ggml_context* ctx, ggml_tensor* x);

// Load vocab/merges KV arrays written by the converters.
bool clip_bpe_load(ClipBpe& bpe, const gguf_context* g, const char* vocab_key,
                   const char* merges_key, int expect_vocab);

// Tokenize `text` into ctx_len token ids exactly like clip.tokenize:
// lowercase -> BPE -> [sot, ..., eot], zero-padded, truncating at ctx_len.
// Returns the number of tokens written (<= ctx_len).
int clip_bpe_tokenize(ClipBpe& bpe, const char* text, int ctx_len,
                      int32_t* tokens);

}  // namespace clip
