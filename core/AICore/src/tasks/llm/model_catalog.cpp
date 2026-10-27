// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Published local LLM / VLM GGUF pairs. The GGUF files are the upstream
// llama.cpp conversions (ggml-org / Qwen Hugging Face repositories); each URL
// resolves an immutable repository revision, and the SHA-256 is the LFS oid of
// that revision (also pinned in asset_digests.h). A projector must be loaded
// with a text model of the same family.

#include <cstring>

#include "aicore/llm_capi.h"

#define SMOLVLM_256M                                                      \
    "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/" \
    "b9e4379657e1450d04d02eec8e345667265b0a00/"
#define SMOLVLM_500M                                                      \
    "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/" \
    "72e986006ef53e37cdd3f6d4241c90b0f01df376/"
#define QWEN25_05B                                                     \
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/" \
    "9217f5db79a29953eb74d5343926648285ec7e67/"

namespace {
const aicore_llm_model_entry kCatalog[] = {
        {"SmolVLM-256M-Instruct-Q8_0.gguf",
         SMOLVLM_256M "SmolVLM-256M-Instruct-Q8_0.gguf",
         "SmolVLM 256M Instruct (Q8_0, vision)", "smolvlm-256m",
         AICORE_LLM_ROLE_MODEL, "q8_0",
         "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
         "2a31195d3769c0b0fd0a4906201666108834848db768af11de1d2cef7cd35e65",
         175054528},
        {"SmolVLM-256M-Instruct-f16.gguf",
         SMOLVLM_256M "SmolVLM-256M-Instruct-f16.gguf",
         "SmolVLM 256M Instruct (F16, vision)", "smolvlm-256m",
         AICORE_LLM_ROLE_MODEL, "f16", "mmproj-SmolVLM-256M-Instruct-f16.gguf",
         "2f0cf33ce293f1a17b094e13e57f7ebd660ad6c0467a5e567a753e333a0c2429",
         327809728},
        {"mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
         SMOLVLM_256M "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
         "SmolVLM 256M projector (Q8_0)", "smolvlm-256m",
         AICORE_LLM_ROLE_MMPROJ, "q8_0", nullptr,
         "7e943f7c53f0382a6fc41b6ee0c2def63ba4fded9ab8ed039cc9e2ab905e0edd",
         103769856},
        {"mmproj-SmolVLM-256M-Instruct-f16.gguf",
         SMOLVLM_256M "mmproj-SmolVLM-256M-Instruct-f16.gguf",
         "SmolVLM 256M projector (F16)", "smolvlm-256m",
         AICORE_LLM_ROLE_MMPROJ, "f16", nullptr,
         "0802360aca1748f112ea510b8ff277c65b1361c8ef30ed89b83c9c7a60d08e96",
         190031616},
        {"SmolVLM-500M-Instruct-Q8_0.gguf",
         SMOLVLM_500M "SmolVLM-500M-Instruct-Q8_0.gguf",
         "SmolVLM 500M Instruct (Q8_0, vision) (recommended)", "smolvlm-500m",
         AICORE_LLM_ROLE_MODEL, "q8_0",
         "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
         "9d4612de6a42214499e301494a3ecc2be0abdd9de44e663bda63f1152fad1bf4",
         436806912},
        {"SmolVLM-500M-Instruct-f16.gguf",
         SMOLVLM_500M "SmolVLM-500M-Instruct-f16.gguf",
         "SmolVLM 500M Instruct (F16, vision)", "smolvlm-500m",
         AICORE_LLM_ROLE_MODEL, "f16", "mmproj-SmolVLM-500M-Instruct-f16.gguf",
         "26c2ee46ef9f8472b3d305a71703153a55f4f2c3baaa12ff5feb566fddb65343",
         820422912},
        {"mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
         SMOLVLM_500M "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
         "SmolVLM 500M projector (Q8_0)", "smolvlm-500m",
         AICORE_LLM_ROLE_MMPROJ, "q8_0", nullptr,
         "d1eb8b6b23979205fdf63703ed10f788131a3f812c7b1f72e0119d5d81295150",
         108783360},
        {"mmproj-SmolVLM-500M-Instruct-f16.gguf",
         SMOLVLM_500M "mmproj-SmolVLM-500M-Instruct-f16.gguf",
         "SmolVLM 500M projector (F16)", "smolvlm-500m",
         AICORE_LLM_ROLE_MMPROJ, "f16", nullptr,
         "4b2fca922e549e9b9608f4ba56f93565dd5220e935ba5570a2a864a0fcc988a6",
         199468800},
        {"qwen2.5-0.5b-instruct-q8_0.gguf",
         QWEN25_05B "qwen2.5-0.5b-instruct-q8_0.gguf",
         "Qwen2.5 0.5B Instruct (Q8_0, text)", "qwen2.5-0.5b",
         AICORE_LLM_ROLE_MODEL, "q8_0", nullptr,
         "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
         675710816},
        {"qwen2.5-0.5b-instruct-fp16.gguf",
         QWEN25_05B "qwen2.5-0.5b-instruct-fp16.gguf",
         "Qwen2.5 0.5B Instruct (F16, text)", "qwen2.5-0.5b",
         AICORE_LLM_ROLE_MODEL, "f16", nullptr,
         "8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc",
         1266425696},
};
constexpr int kCount = int(sizeof(kCatalog) / sizeof(kCatalog[0]));
}  // namespace

int aicore_llm_model_count(void) { return kCount; }
const aicore_llm_model_entry* aicore_llm_model_at(int index) {
    return index >= 0 && index < kCount ? &kCatalog[index] : nullptr;
}
const aicore_llm_model_entry* aicore_llm_model_by_filename(
        const char* filename) {
    if (!filename) return nullptr;
    for (const auto& entry : kCatalog)
        if (std::strcmp(entry.filename, filename) == 0) return &entry;
    return nullptr;
}
// SmolVLM 500M Q8_0 is the vision model upstream llama.cpp gates in
// tools/mtmd/tests.sh (the 256M rows are disabled there).
int aicore_llm_model_default_index(void) { return 4; }
