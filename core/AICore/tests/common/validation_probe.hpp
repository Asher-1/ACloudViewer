#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "aicore/pipeline_timing.h"

namespace aicore::test {

inline uint64_t fnv1aAppend(uint64_t hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline uint64_t fnv1a(const void* data, size_t size) {
    return fnv1aAppend(1469598103934665603ULL, data, size);
}

inline void printValidationResult(const char* task,
                                  const char* device,
                                  uint64_t output_hash,
                                  const aicore_pipeline_timings* timing) {
    const aicore_pipeline_timings empty{};
    const auto& value = timing ? *timing : empty;
    std::printf(
            "{\"suite\":\"aicore-validation\",\"task\":\"%s\","
            "\"device\":\"%s\",\"valid_fields\":%u,"
            "\"preprocess_ms\":%.6f,\"inference_ms\":%.6f,"
            "\"postprocess_ms\":%.6f,\"serialization_ms\":%.6f,"
            "\"e2e_ms\":%.6f,\"output_hash\":\"%016llx\"}\n",
            task, device ? device : "", value.valid_fields, value.preprocess_ms,
            value.inference_ms, value.postprocess_ms, value.serialization_ms,
            value.e2e_ms, static_cast<unsigned long long>(output_hash));
}

}  // namespace aicore::test
