#pragma once


// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: AGPL-3.0
// ----------------------------------------------------------------------------
//
// MobileCLIP2-B TEXT encoder inference on ggml (YOLOE's text tower) —
// in-tree port of ultralytics-ggml cpp_ggml/src/mobileclip_graph.{hpp,cpp}.
//
// Usage:
//   mobileclip::Session* s = mobileclip::create_session("mobileclip2_b-f16.gguf");
//   float embed[512];
//   mobileclip::encode_string(s, "person", embed);  // BPE + encode, end to end
//   mobileclip::free_session(s);
//
// The GGUF (scripts/convert_mobileclip_to_gguf.py) carries the encoder
// weights plus the CLIP SimpleTokenizer vocab/merges, so the runtime accepts
// plain-text class names. Output is the L2-normalised 512-d text feature —
// the pre-reprta embedding that the YOLOE detector graph feeds into its
// reprta projection.

#include "tasks/yolo/yolo_clip_text_graph.hpp"

namespace mobileclip {

// MobileCLIP2-B constants (mirror scripts/convert_mobileclip_to_gguf.py).
constexpr int EMBED_DIM = clip::EMBED_DIM;       // 512
constexpr int TEXT_CTX = clip::TEXT_CTX;         // 77
constexpr int N_LAYERS = 12;
constexpr int N_HEADS = 8;
constexpr int VOCAB_SIZE = clip::VOCAB_SIZE;     // 49408

struct Session {
    ggml_context* wctx = nullptr;  // weight tensor structs (host data)
    yolo::BackendCtx backend;      // PRIVATE CPU-only bundle (never leased;
                                   // see yolo_clip_text_graph.hpp)
    bool owns_cpu_backend = false; // backend.cpu created here (freed here)

    ggml_tensor* embed_table = nullptr;  // token_embedding.weight [49408, 512]
    ggml_tensor* pos_embed = nullptr;    // positional_embedding   [77, 512]
    ggml_tensor* ln_pre_w = nullptr;     // [512]
    ggml_tensor* ln_pre_b = nullptr;     // [512]
    ggml_tensor* proj = nullptr;         // text_projection [512, 512]

    // MobileCLIP block: attention consumes the LayerNormed view of the
    // stream, residuals hit the raw stream, ln_mid sits between the two
    // sub-blocks and ln_post feeds the next block (or the EOS pooling).
    struct Block {
        ggml_tensor* attn_in_w;   // [1536, 512] (qkv combined)
        ggml_tensor* attn_in_b;   // [1536]
        ggml_tensor* attn_out_w;  // [512, 512]
        ggml_tensor* attn_out_b;  // [512]
        ggml_tensor* ln_mid_w;    // [512]
        ggml_tensor* ln_mid_b;    // [512]
        ggml_tensor* mlp_fc_w;    // [2048, 512]
        ggml_tensor* mlp_fc_b;    // [2048]
        ggml_tensor* mlp_proj_w;  // [512, 2048]
        ggml_tensor* mlp_proj_b;  // [512]
        ggml_tensor* ln_post_w;   // [512]
        ggml_tensor* ln_post_b;   // [512]
    };
    Block blocks[N_LAYERS];

    // BPE tokenizer tables (same CLIP SimpleTokenizer as clip-ViT-B-32).
    clip::ClipBpe bpe;

    ggml_cgraph* graph = nullptr;
    ggml_tensor* input_tokens = nullptr;  // int32 [TEXT_CTX]
    ggml_tensor* eot_idx = nullptr;       // int32 [1] EOS position (dynamic)
    ggml_tensor* output_embed = nullptr;  // float [EMBED_DIM]
    ggml_context* gctx = nullptr;         // graph context (freed on destroy)
};

// Create a session from a MobileCLIP GGUF file. Returns nullptr on error.
Session* create_session(const std::string& gguf_path, int threads = 0);

// Free a session previously created with create_session. Safe on NULL.
void free_session(Session* s);

// Encode tokenized text (TEXT_CTX int32 token IDs, zero-padded) into a
// 512-d L2-normalised embedding.
bool encode_tokens(Session* s, const int32_t* tokens, float* embed);

// Convenience: BPE-tokenize a plain string, then encode it.
bool encode_string(Session* s, const char* text, float* embed);

}  // namespace mobileclip
