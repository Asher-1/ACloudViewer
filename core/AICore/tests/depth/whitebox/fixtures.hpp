#pragma once

#include <cstdlib>

namespace aicore_test {
namespace depth {

// Resolve AICORE_TEST_DEPTH_* with optional legacy DA_TEST_* fallback for local harnesses.
inline const char* env_or_legacy(const char* key, const char* legacy = nullptr) {
    if (const char* v = std::getenv(key)) return v;
    if (legacy && (v = std::getenv(legacy))) return v;
    return nullptr;
}

inline const char* gguf() {
    return env_or_legacy("AICORE_TEST_DEPTH_GGUF", "DA_TEST_GGUF");
}
inline const char* baseline() {
    return env_or_legacy("AICORE_TEST_DEPTH_BASELINE", "DA_TEST_BASELINE");
}
inline const char* baseline_mv() {
    return env_or_legacy("AICORE_TEST_DEPTH_BASELINE_MV", "DA_TEST_BASELINE_MV");
}
inline const char* baseline_mv4() {
    return env_or_legacy("AICORE_TEST_DEPTH_BASELINE_MV4", "DA_TEST_BASELINE_MV4");
}
inline const char* baseline_nested() {
    return env_or_legacy("AICORE_TEST_DEPTH_BASELINE_NESTED", "DA_TEST_BASELINE_NESTED");
}
inline const char* baseline_giant() {
    return env_or_legacy("AICORE_TEST_DEPTH_BASELINE_GIANT", "DA_TEST_BASELINE_GIANT");
}
inline const char* baseline_da2() {
    return env_or_legacy("AICORE_TEST_DEPTH_BASELINE_DA2", "DA_TEST_BASELINE_DA2");
}
inline const char* baseline_mono() {
    return env_or_legacy("AICORE_TEST_DEPTH_BASELINE_MONO", "DA_TEST_BASELINE_MONO");
}
inline const char* baseline_native() {
    return env_or_legacy("AICORE_TEST_DEPTH_BASELINE_NATIVE", "DA_TEST_BASELINE_NATIVE");
}
inline const char* baseline_preproc() {
    return env_or_legacy("AICORE_TEST_DEPTH_BASELINE_PREPROC", "DA_TEST_BASELINE_PREPROC");
}
inline const char* baseline_ray_pose() {
    return env_or_legacy("AICORE_TEST_DEPTH_BASELINE_RAY_POSE", "DA_TEST_BASELINE_RAY_POSE");
}
inline const char* baseline_rays() {
    return env_or_legacy("AICORE_TEST_DEPTH_BASELINE_RAYS", "DA_TEST_BASELINE_RAYS");
}
inline const char* gguf_da2() {
    return env_or_legacy("AICORE_TEST_DEPTH_GGUF_DA2", "DA_TEST_GGUF_DA2");
}
inline const char* gguf_mono() {
    return env_or_legacy("AICORE_TEST_DEPTH_GGUF_MONO", "DA_TEST_GGUF_MONO");
}
inline const char* gguf_giant() {
    return env_or_legacy("AICORE_TEST_DEPTH_GGUF_GIANT", "DA_TEST_GGUF_GIANT");
}
inline const char* gguf_metric() {
    return env_or_legacy("AICORE_TEST_DEPTH_GGUF_METRIC", "DA_TEST_GGUF_METRIC");
}
inline const char* gguf_anyview() {
    return env_or_legacy("AICORE_TEST_DEPTH_GGUF_ANYVIEW", "DA_TEST_GGUF_ANYVIEW");
}
inline const char* gguf_aux() {
    return env_or_legacy("AICORE_TEST_DEPTH_GGUF_AUX", "DA_TEST_GGUF_AUX");
}
inline const char* image() {
    return env_or_legacy("AICORE_TEST_DEPTH_IMAGE", "DA_TEST_NATIVE_PNG");
}
inline const char* mono_image() {
    return env_or_legacy("AICORE_TEST_DEPTH_MONO_IMAGE", "DA_TEST_MONO_PNG");
}
inline const char* preproc_image() {
    return env_or_legacy("AICORE_TEST_DEPTH_PREPROC_IMAGE", "DA_TEST_PREPROC_PNG");
}

} // namespace depth
} // namespace aicore_test
