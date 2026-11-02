// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Published VLX-Seek GGUF catalog. The GGUF files are the upstream
// Asher-1/VLX-Seek-GGML conversions published on the
// Asher-1/VLX-Seek-1.5-10B-GGUF Hugging Face repository; the URL resolves an
// immutable repository revision and the SHA-256 is the content digest of
// that revision (also pinned in asset_digests.h).
//
// Role pairing: one family = {wedetect-base-uni-f32 (always f32 — the
// official .pth dtype; quantization changes the proposal set),
// vlx-vision-10b-<precision>, vlx-10b-llm-<precision>}. The q8_0 rows are
// the recommended set (RTX 4090 measurements: steady-state TTFT 1.04x the
// official backend, decode 93 tok/s, ~11 GiB resident vs ~20.5 GiB f16).

#include <cstring>

#include "aicore/vlx_seek_capi.h"

#define VLX_SEEK_REPO_BASE                                             \
    "https://huggingface.co/Asher-1/VLX-Seek-1.5-10B-GGUF/resolve/" \
    "3203c2011434cc8931b403f027e8365eb49071b2/"

namespace {
const aicore_vlx_seek_model_entry kCatalog[] = {
        {"wedetect-base-uni-f32.gguf",
         VLX_SEEK_REPO_BASE "wedetect-base-uni-f32.gguf",
         "WeDetect-Uni proposals (F32, required)", "vlx-seek-1.5-10b",
         AICORE_VLX_SEEK_ROLE_WEDETECT, "f32",
         "21f0e85939cf83260d23d6e00f7095d1c130b1ef174bd16fc61faab91387a345",
         430760800},
        {"wedetect-base-uni-f16.gguf",
         VLX_SEEK_REPO_BASE "wedetect-base-uni-f16.gguf",
         "WeDetect-Uni proposals (F16, smoke only)", "vlx-seek-1.5-10b",
         AICORE_VLX_SEEK_ROLE_WEDETECT, "f16",
         "4cb6c759216cf965064fb34815eeb6bde57af2393067ed0787454ca2a1c728aa",
         217518944},
        {"wedetect-base-uni-q8_0.gguf",
         VLX_SEEK_REPO_BASE "wedetect-base-uni-q8_0.gguf",
         "WeDetect-Uni proposals (Q8_0, smoke only)", "vlx-seek-1.5-10b",
         AICORE_VLX_SEEK_ROLE_WEDETECT, "q8_0",
         "297d6174bea32bf66229b4062cda619d05ccebfde1118084a5e8c7349bdd7627",
         117567584},
        {"vlx-vision-10b-f16.gguf",
         VLX_SEEK_REPO_BASE "vlx-vision-10b-f16.gguf",
         "VLX-Seek 1.5-10B vision stack (F16)", "vlx-seek-1.5-10b",
         AICORE_VLX_SEEK_ROLE_VISION, "f16",
         "1a4ce557fcd45bd88bf83488ab6b3765328cb06c4738506fd8cc8edb23af4ebb",
         2173266400},
        {"vlx-vision-10b-q8_0.gguf",
         VLX_SEEK_REPO_BASE "vlx-vision-10b-q8_0.gguf",
         "VLX-Seek 1.5-10B vision stack (Q8_0, recommended)", "vlx-seek-1.5-10b",
         AICORE_VLX_SEEK_ROLE_VISION, "q8_0",
         "0c9a19352ac91efcb6cd3e4860fd25960dfef52490f51e9d43c70ea0e9aa818a",
         1447540960},
        {"vlx-10b-llm-f16.gguf",
         VLX_SEEK_REPO_BASE "vlx-10b-llm-f16.gguf",
         "VLX-Seek 1.5-10B LLM (F16)", "vlx-seek-1.5-10b",
         AICORE_VLX_SEEK_ROLE_LLM, "f16",
         "00ed394f57515e42a337bc26ee7ab71ac48506f734891de58ccba6924d28c316",
         17918424416},
        {"vlx-10b-llm-q8_0.gguf",
         VLX_SEEK_REPO_BASE "vlx-10b-llm-q8_0.gguf",
         "VLX-Seek 1.5-10B LLM (Q8_0, recommended)", "vlx-seek-1.5-10b",
         AICORE_VLX_SEEK_ROLE_LLM, "q8_0",
         "e0cdb04bcc034971ac97dce715cbbafb945b16bf6f0e366ea79560a74dd13dcb",
         9526288736},
};
constexpr int kCount = int(sizeof(kCatalog) / sizeof(kCatalog[0]));
}  // namespace

int aicore_vlx_seek_model_count(void) { return kCount; }
const aicore_vlx_seek_model_entry* aicore_vlx_seek_model_at(int index) {
    return index >= 0 && index < kCount ? &kCatalog[index] : nullptr;
}
// The q8_0 vision row is the default (recommended) entry; its display name
// carries the visible marker.
int aicore_vlx_seek_model_default_index(void) { return 4; }
const aicore_vlx_seek_model_entry* aicore_vlx_seek_model_by_filename(
        const char* filename) {
    if (!filename) return nullptr;
    for (const auto& entry : kCatalog)
        if (std::strcmp(entry.filename, filename) == 0) return &entry;
    return nullptr;
}
const char* aicore_vlx_seek_model_download_base(void) {
    return VLX_SEEK_REPO_BASE;
}
