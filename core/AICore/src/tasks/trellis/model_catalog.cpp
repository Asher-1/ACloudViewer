// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Published TRELLIS.2 GGUF catalog. This is the only model source of truth
// for both AICore validation and qTrellis. Entries are the Hugging Face LFS
// metadata published by Asher-1/Trellis2-models, including the 2026-08-29 f32
// variants.

#include <cstring>

#include "aicore/trellis_capi.h"

namespace {

static constexpr const char* kDownloadBase =
        "https://huggingface.co/Asher-1/Trellis2-models/resolve/main/";

static constexpr const char* kTrellisLicense =
        "MIT (TRELLIS.2-4B-GGUF / TRELLIS-image-large-GGUF)";
static constexpr const char* kDinoLicense =
        "DINOv3 License (built with DINOv3)";
static constexpr const char* kRmbgLicense =
        "CC BY-NC 4.0 (non-commercial); commercial license from BRIA";

#define TRELLIS_URL(file) \
    "https://huggingface.co/Asher-1/Trellis2-models/resolve/main/" file
#define TRELLIS_ENTRY(file, display, quant, license, role, size, digest) \
    {file, TRELLIS_URL(file), display, quant, license, role, size##ULL, digest}

static const aicore_trellis_model_entry kModels[] = {
        TRELLIS_ENTRY("dino_f16.gguf",
                      "DINOv3 ViT-L/16 f16",
                      "F16 half precision",
                      kDinoLicense,
                      "dino",
                      606992192,
                      "385d8186a38a2328ec740fb2ac1f33f9194d8774efc7ccafd4aa2e51"
                      "cf5f6450"),
        TRELLIS_ENTRY("dino_q8.gguf",
                      "DINOv3 ViT-L/16 q8",
                      "Q8 8-bit quant",
                      kDinoLicense,
                      "dino",
                      323876672,
                      "7da7e92438b0478a10a67d12a1e4439c9a54142d70c073fbe298dda1"
                      "db10de53"),
        TRELLIS_ENTRY("dino_f32.gguf",
                      "DINOv3 ViT-L/16 f32",
                      "F32 exact reference",
                      kDinoLicense,
                      "dino",
                      1212544832,
                      "e024fbd0e8c2906cd85005597197434aac03b383f5eb11987076304b"
                      "33eb8956"),

        TRELLIS_ENTRY("ss_flow_f16.gguf",
                      "Sparse-structure flow f16",
                      "F16 half precision",
                      kTrellisLicense,
                      "ss_flow",
                      2615168864,
                      "1dded5b74237d24e6876a642a26f90b43742e3554418573860f810e3"
                      "bbe61e8c"),
        TRELLIS_ENTRY("ss_flow_q8.gguf",
                      "Sparse-structure flow q8",
                      "Q8 8-bit quant",
                      kTrellisLicense,
                      "ss_flow",
                      1418183264,
                      "a75ab3b3c225bc62b7b33c54fee9d92e936c270ad5579daad4a6c8a3"
                      "919a8d03"),
        TRELLIS_ENTRY("ss_flow_f32.gguf",
                      "Sparse-structure flow f32",
                      "F32 exact reference",
                      kTrellisLicense,
                      "ss_flow",
                      5168762720,
                      "bc2e98e6ee92f5516c04c6e868e0da1c64d388c2c107fb6aa79b07e4"
                      "039c5f07"),
        TRELLIS_ENTRY("ss_dec_f16.gguf",
                      "Occupancy decoder f16",
                      "F16 half precision",
                      kTrellisLicense,
                      "ss_dec",
                      147379616,
                      "9c2210b7ed830fdc8286961a8189878ff5bcfd3bfc83ab4eacee005d"
                      "293d2185"),
        TRELLIS_ENTRY("ss_dec_q8.gguf",
                      "Occupancy decoder q8",
                      "Q8 8-bit quant",
                      kTrellisLicense,
                      "ss_dec",
                      147379616,
                      "fe390843dcd2ca68fdb3d80bae0a2c9992d083b56844cfed3bcdfc20"
                      "17a179ef"),
        TRELLIS_ENTRY("ss_dec_f32.gguf",
                      "Occupancy decoder f32",
                      "F32 exact reference",
                      kTrellisLicense,
                      "ss_dec",
                      294689888,
                      "244748ddc290a2892cab87f994375d8526b6206c5b6ff0422aaabc38"
                      "b806877d"),

        TRELLIS_ENTRY("slat_flow_f16.gguf",
                      "Shape-SLAT flow 512 f16",
                      "F16 half precision",
                      kTrellisLicense,
                      "slat_flow",
                      2615319424,
                      "2f94bad7b1c524ad8c01943bc38fcc0c314e7d482ce896f3c6e96eb6"
                      "e7cec15c"),
        TRELLIS_ENTRY("slat_flow_q8.gguf",
                      "Shape-SLAT flow 512 q8",
                      "Q8 8-bit quant",
                      kTrellisLicense,
                      "slat_flow",
                      1418253184,
                      "fedcc106efed4eb5469af4df8f380004271164a680ff642cd5bc6d61"
                      "4898c92a"),
        TRELLIS_ENTRY("slat_flow_f32.gguf",
                      "Shape-SLAT flow 512 f32",
                      "F32 exact reference",
                      kTrellisLicense,
                      "slat_flow",
                      5169060736,
                      "e76c3559d320e2151f2fae67d392e1e2f69eed163920aeee5c12b2c8"
                      "3edaa56e"),
        TRELLIS_ENTRY("slat_flow_1024_f16.gguf",
                      "Shape-SLAT flow 1024 f16",
                      "F16 half precision",
                      kTrellisLicense,
                      "slat_flow_hr",
                      2630208384,
                      "e4cccf387fb31143eb000213e88b5c820f75cfea660e65914408a732"
                      "9c118249"),
        TRELLIS_ENTRY("slat_flow_1024_q8.gguf",
                      "Shape-SLAT flow 1024 q8",
                      "Q8 8-bit quant",
                      kTrellisLicense,
                      "slat_flow_hr",
                      1418253184,
                      "26577944aed86c270773262b13503a3a341b2d07bd38e19d879474c5"
                      "1a05c29d"),
        TRELLIS_ENTRY("slat_flow_1024_f32.gguf",
                      "Shape-SLAT flow 1024 f32",
                      "F32 exact reference",
                      kTrellisLicense,
                      "slat_flow_hr",
                      5169060736,
                      "554474fd8593f640339b8709dbed6494b396b6d9784d884ab10ae19e"
                      "2a47f637"),
        TRELLIS_ENTRY("shape_dec_f16.gguf",
                      "Shape decoder f16",
                      "F16 half precision",
                      kTrellisLicense,
                      "shape_dec",
                      948745408,
                      "6fe53f1d7763dabf7c8d72bc38f4053d87fde6f65bf17a9d378d27ed"
                      "b39d3530"),
        TRELLIS_ENTRY("shape_dec_f32.gguf",
                      "Shape decoder f32",
                      "F32 exact reference",
                      kTrellisLicense,
                      "shape_dec",
                      1896943680,
                      "2965e10d8d0ec043ba0d6d7cda84e8ebe56a8ca1e63407368e295a8a"
                      "3154cf56"),

        TRELLIS_ENTRY("shape_enc_f16.gguf",
                      "Shape encoder f16",
                      "F16 half precision",
                      kTrellisLicense,
                      "shape_enc",
                      709034048,
                      "3ec80ff580987fcdb9bc594fc8b6fda890d63101ca442eb2b26f5dc3"
                      "15e8696c"),
        TRELLIS_ENTRY("tex_dec_f16.gguf",
                      "Texture decoder f16",
                      "F16 half precision",
                      kTrellisLicense,
                      "tex_dec",
                      948713856,
                      "afd304f4dfcb8c94df851b85519b415b99f04070f7d29de1320c5063"
                      "1b1be4e0"),
        TRELLIS_ENTRY("tex_slat_flow_512_f16.gguf",
                      "Texture flow 512 f16",
                      "F16 half precision",
                      kTrellisLicense,
                      "tex_flow",
                      2615421184,
                      "89a081b7f5487a5b31f03d240e4d959a56db0cc2c46c327230097a25"
                      "54da52ae"),
        TRELLIS_ENTRY("tex_slat_flow_512_q8.gguf",
                      "Texture flow 512 q8",
                      "Q8 8-bit quant",
                      kTrellisLicense,
                      "tex_flow",
                      1418308864,
                      "48f3f023ac24c76fd498ec7914dadbdd644b9d260248b293efcbbd90"
                      "5b75f191"),
        TRELLIS_ENTRY("tex_slat_flow_1024_f16.gguf",
                      "Texture flow 1024 f16",
                      "F16 half precision",
                      kTrellisLicense,
                      "tex_flow_hr",
                      2615421184,
                      "bbb55b0910c7929aac5e0612a9bb15113837a2c674cafb9f0f170eda"
                      "8b5558a8"),
        TRELLIS_ENTRY("tex_slat_flow_1024_q8.gguf",
                      "Texture flow 1024 q8",
                      "Q8 8-bit quant",
                      kTrellisLicense,
                      "tex_flow_hr",
                      1418308864,
                      "24b2cab2429604aa7264e18baa2a86071e6017a74493ce5b5afd2e51"
                      "c8a3cbf5"),

        TRELLIS_ENTRY("rmbg_f32.gguf",
                      "RMBG-2.0 f32",
                      "F32 full precision reference",
                      kRmbgLicense,
                      "rmbg",
                      882846304,
                      "73fa93582743128e392b6e5b6be821e5b67361dcd5a5c0deca0ae407"
                      "7e4c0ddd"),
        TRELLIS_ENTRY("rmbg_f16.gguf",
                      "RMBG-2.0 f16",
                      "F16 half precision",
                      kRmbgLicense,
                      "rmbg",
                      441451648,
                      "50aaf0c7570df97b3767909394d9a63c93effe9f01dc863b17fa69d5"
                      "f76eb8e3"),
        TRELLIS_ENTRY("rmbg_q8.gguf",
                      "RMBG-2.0 q8",
                      "Q8 8-bit quant",
                      kRmbgLicense,
                      "rmbg",
                      258974848,
                      "a2f432614d91057614c59745d40a1770b1a84455f5f853174629b05f"
                      "c7c8e079"),
};

#undef TRELLIS_ENTRY
#undef TRELLIS_URL

constexpr int modelCount() {
    return static_cast<int>(sizeof(kModels) / sizeof(kModels[0]));
}

}  // namespace

AICORE_CAPI int aicore_trellis_model_count(void) { return modelCount(); }

AICORE_CAPI const aicore_trellis_model_entry* aicore_trellis_model_at(
        int index) {
    if (index < 0 || index >= modelCount()) return nullptr;
    return &kModels[index];
}

AICORE_CAPI const aicore_trellis_model_entry* aicore_trellis_model_by_filename(
        const char* filename) {
    if (filename == nullptr || filename[0] == '\0') return nullptr;
    for (const aicore_trellis_model_entry& entry : kModels) {
        if (std::strcmp(entry.filename, filename) == 0) return &entry;
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_trellis_model_download_base(void) {
    return kDownloadBase;
}
