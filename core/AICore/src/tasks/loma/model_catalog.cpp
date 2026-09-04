// Published LoMa GGUF catalog. Artifacts are converted from the pinned
// COLMAP LoMa ONNX graphs; runtime loading and inference remain ggml-only.

#include "aicore/loma_capi.h"

namespace {

constexpr char kDownloadBase[] =
    "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/";

constexpr aicore_loma_model_entry kModels[] = {
    {"loma_detector.f32.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_detector.f32.gguf",
     "LoMa DaD detector (F32, recommended)",
     AICORE_LOMA_MODEL_ROLE_DETECTOR,
     AICORE_LOMA_MODEL_VARIANT_DAD},
    {"loma_descriptor_dedode_b.f32.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_descriptor_dedode_b.f32.gguf",
     "LoMa DeDoDe-B descriptor (F32, B128 compatible)",
     AICORE_LOMA_MODEL_ROLE_DESCRIPTOR,
     AICORE_LOMA_MODEL_VARIANT_DEDODE_B},
    {"loma_descriptor_dedode_g.f32.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_descriptor_dedode_g.f32.gguf",
     "LoMa DeDoDe-G descriptor (F32, recommended)",
     AICORE_LOMA_MODEL_ROLE_DESCRIPTOR,
     AICORE_LOMA_MODEL_VARIANT_DEDODE_G},
    {"loma_matcher_B.f32.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_matcher_B.f32.gguf",
     "LoMa-B matcher (F32, recommended)",
     AICORE_LOMA_MODEL_ROLE_MATCHER,
     AICORE_LOMA_MODEL_VARIANT_MATCHER_B},
    {"loma_matcher_B128.f32.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_matcher_B128.f32.gguf",
     "LoMa-B128 matcher (F32, DeDoDe-B compatible)",
     AICORE_LOMA_MODEL_ROLE_MATCHER,
     AICORE_LOMA_MODEL_VARIANT_MATCHER_B128},
    {"loma_matcher_R.f32.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_matcher_R.f32.gguf",
     "LoMa-R matcher (F32)",
     AICORE_LOMA_MODEL_ROLE_MATCHER,
     AICORE_LOMA_MODEL_VARIANT_MATCHER_R},
    {"loma_matcher_L.f32.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_matcher_L.f32.gguf",
     "LoMa-L matcher (F32)",
     AICORE_LOMA_MODEL_ROLE_MATCHER,
     AICORE_LOMA_MODEL_VARIANT_MATCHER_L},
    {"loma_matcher_G.f32.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_matcher_G.f32.gguf",
     "LoMa-G matcher (F32)",
     AICORE_LOMA_MODEL_ROLE_MATCHER,
     AICORE_LOMA_MODEL_VARIANT_MATCHER_G},
    {"loma_detector.f16.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_detector.f16.gguf",
     "LoMa DaD detector (F16)",
     AICORE_LOMA_MODEL_ROLE_DETECTOR,
     AICORE_LOMA_MODEL_VARIANT_DAD},
    {"loma_descriptor_dedode_b.f16.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_descriptor_dedode_b.f16.gguf",
     "LoMa DeDoDe-B descriptor (F16, B128 compatible)",
     AICORE_LOMA_MODEL_ROLE_DESCRIPTOR,
     AICORE_LOMA_MODEL_VARIANT_DEDODE_B},
    {"loma_descriptor_dedode_g.f16.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_descriptor_dedode_g.f16.gguf",
     "LoMa DeDoDe-G descriptor (F16)",
     AICORE_LOMA_MODEL_ROLE_DESCRIPTOR,
     AICORE_LOMA_MODEL_VARIANT_DEDODE_G},
    {"loma_descriptor_dedode_g.q8_0.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_descriptor_dedode_g.q8_0.gguf",
     "LoMa DeDoDe-G descriptor (Q8_0)",
     AICORE_LOMA_MODEL_ROLE_DESCRIPTOR,
     AICORE_LOMA_MODEL_VARIANT_DEDODE_G},
    {"loma_matcher_B.f16.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_matcher_B.f16.gguf",
     "LoMa-B matcher (F16)",
     AICORE_LOMA_MODEL_ROLE_MATCHER,
     AICORE_LOMA_MODEL_VARIANT_MATCHER_B},
    {"loma_matcher_B.q8_0.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_matcher_B.q8_0.gguf",
     "LoMa-B matcher (Q8_0)",
     AICORE_LOMA_MODEL_ROLE_MATCHER,
     AICORE_LOMA_MODEL_VARIANT_MATCHER_B},
    {"loma_matcher_B128.f16.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_matcher_B128.f16.gguf",
     "LoMa-B128 matcher (F16, DeDoDe-B compatible)",
     AICORE_LOMA_MODEL_ROLE_MATCHER,
     AICORE_LOMA_MODEL_VARIANT_MATCHER_B128},
    {"loma_matcher_B128.q8_0.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_matcher_B128.q8_0.gguf",
     "LoMa-B128 matcher (Q8_0, DeDoDe-B compatible)",
     AICORE_LOMA_MODEL_ROLE_MATCHER,
     AICORE_LOMA_MODEL_VARIANT_MATCHER_B128},
    {"loma_matcher_R.f16.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_matcher_R.f16.gguf",
     "LoMa-R matcher (F16)",
     AICORE_LOMA_MODEL_ROLE_MATCHER,
     AICORE_LOMA_MODEL_VARIANT_MATCHER_R},
    {"loma_matcher_R.q8_0.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_matcher_R.q8_0.gguf",
     "LoMa-R matcher (Q8_0)",
     AICORE_LOMA_MODEL_ROLE_MATCHER,
     AICORE_LOMA_MODEL_VARIANT_MATCHER_R},
    {"loma_matcher_L.f16.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_matcher_L.f16.gguf",
     "LoMa-L matcher (F16)",
     AICORE_LOMA_MODEL_ROLE_MATCHER,
     AICORE_LOMA_MODEL_VARIANT_MATCHER_L},
    {"loma_matcher_L.q8_0.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_matcher_L.q8_0.gguf",
     "LoMa-L matcher (Q8_0)",
     AICORE_LOMA_MODEL_ROLE_MATCHER,
     AICORE_LOMA_MODEL_VARIANT_MATCHER_L},
    {"loma_matcher_G.f16.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_matcher_G.f16.gguf",
     "LoMa-G matcher (F16)",
     AICORE_LOMA_MODEL_ROLE_MATCHER,
     AICORE_LOMA_MODEL_VARIANT_MATCHER_G},
    {"loma_matcher_G.q8_0.gguf",
     "https://github.com/Asher-1/cloudViewer_downloads/releases/download/LoMa_GUFF/"
     "loma_matcher_G.q8_0.gguf",
     "LoMa-G matcher (Q8_0)",
     AICORE_LOMA_MODEL_ROLE_MATCHER,
     AICORE_LOMA_MODEL_VARIANT_MATCHER_G},
};

constexpr int kModelCount = static_cast<int>(sizeof(kModels) / sizeof(kModels[0]));

}  // namespace

AICORE_CAPI int aicore_loma_model_count(void) { return kModelCount; }

AICORE_CAPI int aicore_loma_model_default_index(
    const aicore_loma_model_role role) {
  for (int index = 0; index < kModelCount; ++index) {
    if (kModels[index].role == role) return index;
  }
  return -1;
}

AICORE_CAPI const aicore_loma_model_entry* aicore_loma_model_at(
    const int index) {
  return index >= 0 && index < kModelCount ? &kModels[index] : nullptr;
}

AICORE_CAPI const aicore_loma_model_entry* aicore_loma_model_by_role(
    const aicore_loma_model_role role) {
  const int index = aicore_loma_model_default_index(role);
  return index >= 0 ? &kModels[index] : nullptr;
}

AICORE_CAPI const aicore_loma_model_entry* aicore_loma_model_by_variant(
    const aicore_loma_model_variant variant) {
  for (int index = 0; index < kModelCount; ++index) {
    if (kModels[index].variant == variant) return &kModels[index];
  }
  return nullptr;
}

AICORE_CAPI const char* aicore_loma_model_download_base(void) {
  return kDownloadBase;
}
