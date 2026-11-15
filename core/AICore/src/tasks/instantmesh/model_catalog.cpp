// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Published InstantMesh GGUF catalog. This is the only model source of truth
// for both AICore validation and the qInstantMesh plugin. Entries are the
// Hugging Face LFS metadata published by Asher-1/InstantMesh-GUFF.
//
// RMBG models are intentionally NOT part of this catalog: the in-tree RMBG
// task catalog owns the identical LFS objects (single owner per process).

#include <cstring>

#include "aicore/instantmesh_capi.h"

namespace {

static constexpr const char* kDownloadBase =
        "https://huggingface.co/Asher-1/InstantMesh-GUFF/resolve/main/";

static constexpr const char* kInstantMeshLicense =
        "Apache-2.0 (TencentARC/InstantMesh weights, ggml conversion)";
static constexpr const char* kZero123License =
        "Apache-2.0 (Zero123++ stabilityai/zero123plus-v1.2 weights, ggml "
        "conversion)";

#define IM_URL(file) \
    "https://huggingface.co/Asher-1/InstantMesh-GUFF/resolve/main/" file
#define IM_ENTRY(file, display, quant, license, role, size, digest) \
    {file, IM_URL(file), display, quant, license, role, size##ULL, digest}

static const aicore_instantmesh_model_entry kModels[] = {
        IM_ENTRY("dino_f16.gguf",
                 "DINO ViT-B/16 f16",
                 "F16 half precision (recommended)",
                 kInstantMeshLicense,
                 "dino",
                 230145408,
                 "ab09647993c954e9281a595a410839e60012f75cc44b5272a6a399d5610"
                 "dd88f"),
        IM_ENTRY("dino_q8.gguf",
                 "DINO ViT-B/16 q8",
                 "Q8 8-bit quant",
                 kInstantMeshLicense,
                 "dino",
                 123424128,
                 "d36498038695ca9f15a261a1720ba3b345d8e7fd93659533f242c9f3539"
                 "69b03"),
        IM_ENTRY("dino_f32.gguf",
                 "DINO ViT-B/16 f32",
                 "F32 exact reference",
                 kInstantMeshLicense,
                 "dino",
                 459021696,
                 "388248c31e86620065abd3716ce1cd24737d009f0645fbdb3dd78c78eb7"
                 "cfe85"),

        IM_ENTRY("lrm_transformer_f16.gguf",
                 "TriplaneTransformer f16",
                 "F16 half precision (recommended)",
                 kInstantMeshLicense,
                 "transformer",
                 527786304,
                 "f49a3fa0b12e831bc77644b4386fe0dab9ca961cd99df83b0a3f0bd5a7"
                 "d9fdea"),
        IM_ENTRY("lrm_transformer_q8.gguf",
                 "TriplaneTransformer q8",
                 "Q8 8-bit quant",
                 kInstantMeshLicense,
                 "transformer",
                 283992384,
                 "3e0b458c43bd51fd62e5d3131ef7f1e36c5da3b8056952fe433b520127"
                 "05d1a3"),
        IM_ENTRY("lrm_transformer_f32.gguf",
                 "TriplaneTransformer f32",
                 "F32 exact reference",
                 kInstantMeshLicense,
                 "transformer",
                 1054826816,
                 "6a644ce6c4e99fa9e78fe39b71c649a86646465bd06e3045782a980056c"
                 "9b03f"),

        IM_ENTRY("synthesizer_f16.gguf",
                 "OSGDecoder f16",
                 "F16 half precision (recommended)",
                 kInstantMeshLicense,
                 "synthesizer",
                 412384,
                 "712ac79db8c56ac0c69bb283b76adf267ea5220fad7befef812ce1f8fc2"
                 "83521"),
        IM_ENTRY("synthesizer_q8.gguf",
                 "OSGDecoder q8",
                 "Q8 8-bit quant",
                 kInstantMeshLicense,
                 "synthesizer",
                 264864,
                 "e1419a06dbadb2be4d50ae7aece54c5aefc09a30ad5c44452e34bbff8ec"
                 "e691b"),
        IM_ENTRY("synthesizer_f32.gguf",
                 "OSGDecoder f32",
                 "F32 exact reference",
                 kInstantMeshLicense,
                 "synthesizer",
                 819424,
                 "b6396e6fd1f78f1ab2f167e41678e6e3ce3f13e97171ea75c514b4ac1ed"
                 "fc39d"),

        IM_ENTRY("zero123pp_cond_f32.gguf",
                 "Zero123++ CLIP-Vision + constants f32",
                 "F32 (single published precision)",
                 kZero123License,
                 "cond",
                 2528739904,
                 "1c810da58151c627d88c19e38eb605634a73c681cc8f669a5e25a87cc94"
                 "ec128"),
        IM_ENTRY("zero123pp_unet_f16.gguf",
                 "Zero123++ UNet f16",
                 "F16 half precision (recommended)",
                 kZero123License,
                 "unet",
                 1732768320,
                 "983602d7bfd802083627a32895f266a96e5363d609a7d8b763bcd4e9772"
                 "95222"),
        IM_ENTRY("zero123pp_unet_f32.gguf",
                 "Zero123++ UNet f32",
                 "F32 exact reference",
                 kZero123License,
                 "unet",
                 3463702080,
                 "f07135c6474498c21796cd7baade1b5681f77c420321a74a8d3c67f5413"
                 "71d3e"),
        IM_ENTRY("zero123pp_vae_f16.gguf",
                 "Zero123++ VAE f16",
                 "F16 half precision (recommended)",
                 kZero123License,
                 "vae",
                 167459584,
                 "b91feec6ab803d9b7a19d1b21b6ad9ad1220e575d1def4f94bf0ddb3930"
                 "6baa1"),
        IM_ENTRY("zero123pp_vae_f32.gguf",
                 "Zero123++ VAE f32",
                 "F32 exact reference",
                 kZero123License,
                 "vae",
                 334635424,
                 "6399f42fdf2280d518d46b5e0f6723ca9b5b4571d86abef0e5ad2c84d67"
                 "218f4"),
};

}  // namespace

extern "C" {

AICORE_CAPI int aicore_instantmesh_model_count(void) {
    return (int)(sizeof(kModels) / sizeof(kModels[0]));
}

AICORE_CAPI const aicore_instantmesh_model_entry* aicore_instantmesh_model_at(
        int index) {
    if (index < 0 || index >= aicore_instantmesh_model_count()) return nullptr;
    return &kModels[index];
}

AICORE_CAPI const aicore_instantmesh_model_entry*
aicore_instantmesh_model_by_filename(const char* filename) {
    if (!filename || !filename[0]) return nullptr;
    for (int i = 0; i < aicore_instantmesh_model_count(); ++i) {
        if (std::strcmp(kModels[i].filename, filename) == 0) return &kModels[i];
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_instantmesh_model_download_base(void) {
    return kDownloadBase;
}

}  // extern "C"
