// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <stdint.h>

#define AICORE_PIPELINE_TIMINGS_ABI_VERSION 1u

enum aicore_pipeline_timing_field {
    AICORE_TIMING_PREPROCESS = 1u << 0,
    AICORE_TIMING_INFERENCE = 1u << 1,
    AICORE_TIMING_POSTPROCESS = 1u << 2,
    AICORE_TIMING_SERIALIZATION = 1u << 3,
    AICORE_TIMING_E2E = 1u << 4
};

/** Common wall-clock timing contract for every AICore pipeline.
 *
 * preprocess_ms: API input validation, decode/resize/normalize and graph-input
 * preparation before backend execution.
 * inference_ms: backend upload, graph execution, synchronization and the
 * readback required to expose native outputs.
 * postprocess_ms: native typed-result construction after backend execution.
 * serialization_ms: optional compatibility serialization/encoding only.
 * e2e_ms: API entry until the native result is ready; excludes optional
 * serialization when the pipeline can expose it independently.
 *
 * Consumers must inspect valid_fields. A stage that cannot be measured at a
 * truthful boundary is left at zero and its bit is clear.
 */
typedef struct aicore_pipeline_timings {
    uint32_t abi_version;
    uint32_t valid_fields;
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double serialization_ms;
    double e2e_ms;
} aicore_pipeline_timings;
