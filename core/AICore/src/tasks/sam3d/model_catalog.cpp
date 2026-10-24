// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Single source of truth for the published SAM 3D Objects GGUF inventory
// (https://huggingface.co/Asher-1/SAM_3D_OBJECTS_GGUF). Sizes and SHA-256
// values are the Hugging Face LFS metadata of the files at import time; the
// one-click validation gate verifies digests before inference. qSam3d must
// read model metadata only through the aicore_sam3d_model_* accessors and
// never maintain a second table.
//
// slat_decoder_gs_4 (the stride-4 Gaussian-decoder variant) is published on
// HF but deliberately absent from this runtime catalog: nothing consumes it,
// and an unconsumed row would fail the one-click coverage audit (see the
// note beside the slat_decoder_gs entries).

#include "model_catalog.hpp"

#include <cstring>
#include <vector>

#include "common/data_root_util.hpp"

namespace aicore {
namespace {

constexpr const char* kSam3dDownloadBase =
        "https://huggingface.co/Asher-1/SAM_3D_OBJECTS_GGUF/resolve/main/";

constexpr const char* kSam3dLicense =
        "SAM 3D Objects license (Meta); weights inherit the upstream license";

struct Sam3dCatalogEntry {
    const char* filename;
    const char* display_name;
    const char* quant_note;
    const char* role;
    int64_t size_bytes;
    const char* sha256;
};

const Sam3dCatalogEntry kSam3dModels[] = {
        {"moge_vitl-f16.gguf", "MoGe ViT-L (F16 GGUF)", "f16", "moge",
         629144832,
         "0286470d5e2ee903d39e37efede072f553b72503bed958b2dd07cd48eed87048"},
        {"ss_generator-f16.gguf",
         "SS generator - condition encoder, structure diffusion and pose (F16)",
         "f16", "ss_generator", 3200717760,
         "8d04e6a0797fede96fac57cba4180e19b2154af2440f843109a596184e758627"},
        {"ss_generator-f32.gguf",
         "SS generator - condition encoder, structure diffusion and pose (F32)",
         "f32", "ss_generator", 6394703808,
         "93ed33a2bd50268a71c2f1df0eda691146c503d3a4afa3fd9b0a28db4a5829fc"},
        {"ss_generator-q8_0.gguf",
         "SS generator - condition encoder, structure diffusion and pose "
         "(Q8_0, "
         "cemb.* kept F16)",
         "q8_0 (DINO/PointPatch/fuser weights F16)", "ss_generator", 2297714208,
         "0ebb65088076191ccddd075fbaab80ef63de8760386462f1e929daf5b68b00f2"},
        {"ss_generator-q4_k.gguf",
         "SS generator - condition encoder, structure diffusion and pose "
         "(Q4_K)",
         "q4_k", "ss_generator", 2006933344,
         "b658beeaee87e338761bc9507bd4d090045bd3ba784abf20132749f6e0200990"},
        {"ss_decoder-f16.gguf", "SS decoder - sparse support (F16)", "f16",
         "ss_decoder", 147379456,
         "a003c036b5f8eb5038597a62259df4b2f17b99fd60e5ac0592422fa97c196393"},
        {"ss_decoder-f32.gguf", "SS decoder - sparse support (F32)", "f32",
         "ss_decoder", 294689728,
         "04e4b866b63050e03bd557506bb59b291e0a82c97d8da14dce75e36ed714da7c"},
        {"ss_decoder-q8_0.gguf", "SS decoder - sparse support (Q8_0)", "q8_0",
         "ss_decoder", 78431456,
         "4e4910056a399d55a832a759a11f06c7a3b9c2a543e2b8dc48df90919f7c8f40"},
        {"ss_decoder-q4_k.gguf", "SS decoder - sparse support (Q4_K)", "q4_k",
         "ss_decoder", 45634816,
         "5de992845dc979067210052b49606a6e321fe300f588041b3be402cc508caad1"},
        {"slat_generator-f16.gguf",
         "SLat generator - structured latent diffusion (F16)", "f16",
         "slat_generator", 2455619616,
         "6b3566359b0317a9987539ae43ff7f2245feac2ffe327da98313f50ab60b283b"},
        {"slat_generator-f32.gguf",
         "SLat generator - structured latent diffusion (F32)", "f32",
         "slat_generator", 4906037280,
         "dbf997e0d4897d8e9eca80f267be648845d0c003babf68576eeef0b69a8b2de4"},
        {"slat_generator-q8_0.gguf",
         "SLat generator - structured latent diffusion (Q8_0)", "q8_0",
         "slat_generator", 1308116256,
         "251274fb5c4db62b8759c598376d750db9120b04fd943cdee5c7425509a8fec4"},
        {"slat_generator-q4_k.gguf",
         "SLat generator - structured latent diffusion (Q4_K)", "q4_k",
         "slat_generator", 708061376,
         "2ccc9cf41681989a1141a342b837c8fd79898da5057e0abb33e0541a7e0db107"},
        {"slat_decoder_gs-f16.gguf", "Gaussian decoder (F16)", "f16",
         "slat_decoder_gs", 171614944,
         "3ee8583c47530a96d1826b526118f2fbd3d98d8f48cc66d22f053d39d46fbed1"},
        {"slat_decoder_gs-f32.gguf", "Gaussian decoder (F32)", "f32",
         "slat_decoder_gs", 341484160,
         "364f20e4bf8de55f81bd2ccccd124cca5be5af1ffa47cff852147f93d75d19b7"},
        {"slat_decoder_gs-q8_0.gguf", "Gaussian decoder (Q8_0)", "q8_0",
         "slat_decoder_gs", 91988704,
         "a54a04e18fbc3957f986f114da9f953cfbfdda280fac4e1a0b61a74e811679b0"},
        {"slat_decoder_gs-q4_k.gguf", "Gaussian decoder (Q4_K)", "q4_k",
         "slat_decoder_gs", 49521376,
         "118ac7225383dc4d73201050aa4ac265b9a2021aa217fea03d04e609623b15ed"},
        // slat_decoder_gs_4 (stride-4 Gaussian decoder, published on HF) is
        // deliberately absent from the runtime catalog: no native pipeline or
        // manifest scenario consumes it, and an unconsumed catalog row would
        // fail the one-click coverage audit. Register it together with its
        // consumer when a stride-4 path is wired up.
        {"slat_decoder_mesh-f16.gguf",
         "Mesh decoder - FlexiCubes surface features (F16)", "f16",
         "slat_decoder_mesh", 181900768,
         "026b30cea50e7c4e0894e0a2b6c38534c62e99ab6e76e5248096c7e9b9a1c858"},
        {"slat_decoder_mesh-f32.gguf",
         "Mesh decoder - FlexiCubes surface features (F32)", "f32",
         "slat_decoder_mesh", 363715680,
         "2ecdab81b4a0e50c85b5d9bc1ab663db82266c85339eef42a2fabbdd492d1b2e"},
        {"slat_decoder_mesh-q8_0.gguf",
         "Mesh decoder - FlexiCubes surface features (Q8_0)", "q8_0",
         "slat_decoder_mesh", 96880352,
         "fded9cfcd90163b1c619c5c9b2c51455d8f0e09aa8b7183698ff275e4a2b336a"},
        {"slat_decoder_mesh-q4_k.gguf",
         "Mesh decoder - FlexiCubes surface features (Q4_K)", "q4_k",
         "slat_decoder_mesh", 51477600,
         "a5b7b3e5691596d0e842bf0bab610a77add27a1cc0e68fb1093fd1137c68d287"},
};

const int kSam3dModelCount =
        static_cast<int>(sizeof(kSam3dModels) / sizeof(kSam3dModels[0]));

}  // namespace

const char* sam3d_model_download_base() { return kSam3dDownloadBase; }

const char* sam3d_model_license_note() { return kSam3dLicense; }

int sam3d_model_count() { return kSam3dModelCount; }

const aicore_sam3d_model_entry* sam3d_model_at(int index) {
    if (index < 0 || index >= kSam3dModelCount) return nullptr;
    static const std::vector<std::string> kUrls = [] {
        std::vector<std::string> urls;
        urls.reserve(kSam3dModelCount);
        for (const Sam3dCatalogEntry& e : kSam3dModels) {
            urls.push_back(std::string(kSam3dDownloadBase) + e.filename);
        }
        return urls;
    }();
    static thread_local aicore_sam3d_model_entry out;
    const Sam3dCatalogEntry& e = kSam3dModels[index];
    out.filename = e.filename;
    out.download_url = kUrls[static_cast<size_t>(index)].c_str();
    out.display_name = e.display_name;
    out.quant_note = e.quant_note;
    out.license_note = kSam3dLicense;
    out.role = e.role;
    out.size_bytes = e.size_bytes;
    out.sha256 = e.sha256;
    return &out;
}

const aicore_sam3d_model_entry* sam3d_model_by_filename(const char* filename) {
    if (filename == nullptr) return nullptr;
    for (int i = 0; i < kSam3dModelCount; ++i) {
        if (std::strcmp(kSam3dModels[i].filename, filename) == 0) {
            return sam3d_model_at(i);
        }
    }
    return nullptr;
}

std::string sam3d_model_cache_dir() {
    return extract_model_dir("sam3d_models");
}

}  // namespace aicore
