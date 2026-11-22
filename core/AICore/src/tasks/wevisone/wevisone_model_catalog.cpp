// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstring>

#include "aicore/wevisone_capi.h"

#define WEVISONE_REPO_BASE                                     \
    "https://huggingface.co/Asher-1/WeVisOne-VL-GGML/resolve/" \
    "ec7bf6528100a17ac97269e1a7d94d35d9e0c792/"

namespace {
const aicore_wevisone_model_entry kCatalog[] = {
        {"wevisone-vlm-2b-f16.gguf",
         WEVISONE_REPO_BASE "wevisone-vlm-2b-f16.gguf",
         "WeVisOne-VL 2B VLM (F16)", "2b", AICORE_WEVISONE_ROLE_VLM, "f16",
         "bcaa9833440753f18117a919355c6d2093ca9435d7173c90b5535471953f5c60",
         4075786432},
        {"wevisone-vlm-2b-q8_0.gguf",
         WEVISONE_REPO_BASE "wevisone-vlm-2b-q8_0.gguf",
         "WeVisOne-VL 2B VLM (Q8_0, recommended)", "2b",
         AICORE_WEVISONE_ROLE_VLM, "q8_0",
         "0888776e7f0b0ab15fd8f7e1d3668a48346a64bba759530afa8a2c3610311d3a",
         2168289472},
        {"wevisone-vlm-2b-q4_K_M.gguf",
         WEVISONE_REPO_BASE "wevisone-vlm-2b-q4_K_M.gguf",
         "WeVisOne-VL 2B VLM (Q4_K_M)", "2b", AICORE_WEVISONE_ROLE_VLM,
         "q4_K_M",
         "88dda10738d3b65efa99506402f1c268b3c531e85b402f5f3f6b1a0f405744e4",
         1284558656},
        {"mmproj-wevisone-vlm-2b-f16.gguf",
         WEVISONE_REPO_BASE "mmproj-wevisone-vlm-2b-f16.gguf",
         "WeVisOne-VL 2B ViT projector (F16, required)", "2b",
         AICORE_WEVISONE_ROLE_MMPROJ, "f16",
         "f18d6af807569e258b56e2e297b9b879ab65995215852353185f8ef4c1b3058b",
         819395168},
        {"wevisone-mask-decoder-2b-f32.gguf",
         WEVISONE_REPO_BASE "wevisone-mask-decoder-2b-f32.gguf",
         "WeVisOne-VL 2B pixel decoder (F32, parity tier)", "2b",
         AICORE_WEVISONE_ROLE_DECODER, "f32",
         "6cb6152caa4ce6cf764b39a605b61145d8e1902bbfad0460afef0a1ee1843d4c",
         38069312},
        {"wevisone-mask-decoder-2b-f16.gguf",
         WEVISONE_REPO_BASE "wevisone-mask-decoder-2b-f16.gguf",
         "WeVisOne-VL 2B pixel decoder (F16)", "2b",
         AICORE_WEVISONE_ROLE_DECODER, "f16",
         "2f9f5fab6f45a74599f26e774aed045f9a1c3a9588e90a894c99961924197be3",
         19046400},
        {"wevisone-vlm-4b-f16.gguf",
         WEVISONE_REPO_BASE "wevisone-vlm-4b-f16.gguf",
         "WeVisOne-VL 4B VLM (F16)", "4b", AICORE_WEVISONE_ROLE_VLM, "f16",
         "21b5ac710912d0a128ad1a27ecc9a1f24d4afdd51aeff3056f39e06474ef12a9",
         8836828448},
        {"wevisone-vlm-4b-q8_0.gguf",
         WEVISONE_REPO_BASE "wevisone-vlm-4b-q8_0.gguf",
         "WeVisOne-VL 4B VLM (Q8_0)", "4b", AICORE_WEVISONE_ROLE_VLM, "q8_0",
         "cf42a6e46b848567fa356591ab0c8289dc16cdec29735f7634dd157bf39f891c",
         4697730848},
        {"wevisone-vlm-4b-q4_K_M.gguf",
         WEVISONE_REPO_BASE "wevisone-vlm-4b-q4_K_M.gguf",
         "WeVisOne-VL 4B VLM (Q4_K_M)", "4b", AICORE_WEVISONE_ROLE_VLM,
         "q4_K_M",
         "7fc123cb5ad7f42630032d4bf16f5ec0e4991c756d2eeadc2eac2203233b4073",
         2718714688},
        {"mmproj-wevisone-vlm-4b-f16.gguf",
         WEVISONE_REPO_BASE "mmproj-wevisone-vlm-4b-f16.gguf",
         "WeVisOne-VL 4B ViT projector (F16, required)", "4b",
         AICORE_WEVISONE_ROLE_MMPROJ, "f16",
         "04026d60c582bc25658fe9f2968c33a269e371f0e752a7562c671390d9e8a143",
         836180576},
        {"wevisone-mask-decoder-4b-f32.gguf",
         WEVISONE_REPO_BASE "wevisone-mask-decoder-4b-f32.gguf",
         "WeVisOne-VL 4B pixel decoder (F32, parity tier)", "4b",
         AICORE_WEVISONE_ROLE_DECODER, "f32",
         "76e0ca93538e3cf562c314732936c49f7626ebb8133beff53b4f44f239f71934",
         38462528},
        {"wevisone-mask-decoder-4b-f16.gguf",
         WEVISONE_REPO_BASE "wevisone-mask-decoder-4b-f16.gguf",
         "WeVisOne-VL 4B pixel decoder (F16)", "4b",
         AICORE_WEVISONE_ROLE_DECODER, "f16",
         "2a4bed7f92a5723ea8c1be74b71176f8563fbce3f82d3f1358b699e0a59e9f9f",
         19243008},
};
constexpr int kCount = int(sizeof(kCatalog) / sizeof(kCatalog[0]));
}  // namespace

int aicore_wevisone_model_count(void) { return kCount; }
const aicore_wevisone_model_entry* aicore_wevisone_model_at(int index) {
    return index >= 0 && index < kCount ? &kCatalog[index] : nullptr;
}
// The 2b q8_0 VLM row is the default (recommended) entry; its display name
// carries the visible marker.
int aicore_wevisone_model_default_index(void) { return 1; }
const aicore_wevisone_model_entry* aicore_wevisone_model_by_filename(
        const char* filename) {
    if (!filename) return nullptr;
    for (const auto& entry : kCatalog)
        if (std::strcmp(entry.filename, filename) == 0) return &entry;
    return nullptr;
}
const char* aicore_wevisone_model_download_base(void) {
    return WEVISONE_REPO_BASE;
}