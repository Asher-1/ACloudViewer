// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdint>

#include "aicore/aliked_capi.h"
#include "aicore/deeplsd_capi.h"
#include "aicore/depth_capi.h"
#include "aicore/facedetect_capi.h"
#include "aicore/gaussian_capi.h"
#include "aicore/lightglue_capi.h"
#include "aicore/rfdetr_capi.h"
#include "aicore/rmbg_capi.h"
#include "aicore/sam3_capi.h"
#include "aicore/trellis_capi.h"
#include "aicore/yolo_capi.h"

#define CHECK(expr)          \
    do {                     \
        if (!(expr)) {       \
            return __LINE__; \
        }                    \
    } while (false)

int main() {
    static_assert(AICORE_PIPELINE_TIMINGS_ABI_VERSION == 1u);
    static_assert(AICORE_TIMING_PREPROCESS == (1u << 0));
    static_assert(AICORE_TIMING_INFERENCE == (1u << 1));
    static_assert(AICORE_TIMING_POSTPROCESS == (1u << 2));
    static_assert(AICORE_TIMING_SERIALIZATION == (1u << 3));
    static_assert(AICORE_TIMING_E2E == (1u << 4));

    aicore_pipeline_timings out{};
    CHECK(aicore_aliked_last_pipeline_timings(nullptr, &out) == -1);
    CHECK(aicore_deeplsd_last_pipeline_timings(nullptr, &out) == -1);
    CHECK(aicore_depth_last_pipeline_timings(nullptr, &out) == -1);
    CHECK(aicore_facedetect_last_pipeline_timings(nullptr, &out) == -1);
    CHECK(aicore_gaussian_last_pipeline_timings(nullptr, &out) == -1);
    CHECK(aicore_lightglue_last_pipeline_timings(nullptr, &out) == -1);
    CHECK(aicore_rfdetr_last_pipeline_timings(nullptr, &out) == -1);
    CHECK(aicore_rmbg_last_pipeline_timings(nullptr, &out) == -1);
    CHECK(aicore_sam3_last_pipeline_timings(nullptr, &out) == -1);
    CHECK(aicore_sam3_tracker_last_pipeline_timings(nullptr, &out) == -1);
    CHECK(aicore_trellis_last_pipeline_timings(nullptr, &out) == -1);
    CHECK(aicore_yolo_last_pipeline_timings(nullptr, &out) == -1);

    CHECK(aicore_depth_last_pipeline_timings(nullptr, nullptr) == -1);
    return 0;
}
