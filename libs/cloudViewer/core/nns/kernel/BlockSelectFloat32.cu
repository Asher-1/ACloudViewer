// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/nns/kernel/BlockSelectImpl.cuh"

namespace cloudViewer {
namespace core {
BLOCK_SELECT_IMPL(float, int32_t, true, 1, 1);
BLOCK_SELECT_IMPL(float, int32_t, false, 1, 1);

BLOCK_SELECT_IMPL(float, int32_t, true, 32, 2);
BLOCK_SELECT_IMPL(float, int32_t, false, 32, 2);

BLOCK_SELECT_IMPL(float, int32_t, true, 64, 3);
BLOCK_SELECT_IMPL(float, int32_t, false, 64, 3);

BLOCK_SELECT_IMPL(float, int32_t, true, 128, 3);
BLOCK_SELECT_IMPL(float, int32_t, false, 128, 3);

BLOCK_SELECT_IMPL(float, int32_t, true, 256, 4);
BLOCK_SELECT_IMPL(float, int32_t, false, 256, 4);

BLOCK_SELECT_IMPL(float, int32_t, true, 512, 8);
BLOCK_SELECT_IMPL(float, int32_t, false, 512, 8);

BLOCK_SELECT_IMPL(float, int32_t, true, 1024, 8);
BLOCK_SELECT_IMPL(float, int32_t, false, 1024, 8);

#if GPU_MAX_SELECTION_K >= 2048
BLOCK_SELECT_IMPL(float, int32_t, true, 2048, 8);
BLOCK_SELECT_IMPL(float, int32_t, false, 2048, 8);
#endif

BLOCK_SELECT_IMPL(float, int64_t, true, 1, 1);
BLOCK_SELECT_IMPL(float, int64_t, false, 1, 1);

BLOCK_SELECT_IMPL(float, int64_t, true, 32, 2);
BLOCK_SELECT_IMPL(float, int64_t, false, 32, 2);

BLOCK_SELECT_IMPL(float, int64_t, true, 64, 3);
BLOCK_SELECT_IMPL(float, int64_t, false, 64, 3);

BLOCK_SELECT_IMPL(float, int64_t, true, 128, 3);
BLOCK_SELECT_IMPL(float, int64_t, false, 128, 3);

BLOCK_SELECT_IMPL(float, int64_t, true, 256, 4);
BLOCK_SELECT_IMPL(float, int64_t, false, 256, 4);

BLOCK_SELECT_IMPL(float, int64_t, true, 512, 8);
BLOCK_SELECT_IMPL(float, int64_t, false, 512, 8);

BLOCK_SELECT_IMPL(float, int64_t, true, 1024, 8);
BLOCK_SELECT_IMPL(float, int64_t, false, 1024, 8);

#if GPU_MAX_SELECTION_K >= 2048
BLOCK_SELECT_IMPL(float, int64_t, true, 2048, 8);
BLOCK_SELECT_IMPL(float, int64_t, false, 2048, 8);
#endif

void runBlockSelectPair(cudaStream_t stream,
                        float* inK,
                        int32_t* inV,
                        float* outK,
                        int32_t* outV,
                        bool dir,
                        int k,
                        int dim,
                        int num_points) {
    CLOUDVIEWER_ASSERT(k <= GPU_MAX_SELECTION_K);

    if (dir) {
        if (k == 1) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, true, 1);
        } else if (k <= 32) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, true, 32);
        } else if (k <= 64) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, true, 64);
        } else if (k <= 128) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, true, 128);
        } else if (k <= 256) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, true, 256);
        } else if (k <= 512) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, true, 512);
        } else if (k <= 1024) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, true, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, true, 2048);
#endif
        }
    } else {
        if (k == 1) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, false, 1);
        } else if (k <= 32) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, false, 32);
        } else if (k <= 64) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, false, 64);
        } else if (k <= 128) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, false, 128);
        } else if (k <= 256) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, false, 256);
        } else if (k <= 512) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, false, 512);
        } else if (k <= 1024) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, false, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BLOCK_SELECT_PAIR_CALL(float, int32_t, false, 2048);
#endif
        }
    }
}

void runBlockSelectPair(cudaStream_t stream,
                        float* inK,
                        int64_t* inV,
                        float* outK,
                        int64_t* outV,
                        bool dir,
                        int k,
                        int dim,
                        int num_points) {
    CLOUDVIEWER_ASSERT(k <= GPU_MAX_SELECTION_K);

    if (dir) {
        if (k == 1) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, true, 1);
        } else if (k <= 32) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, true, 32);
        } else if (k <= 64) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, true, 64);
        } else if (k <= 128) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, true, 128);
        } else if (k <= 256) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, true, 256);
        } else if (k <= 512) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, true, 512);
        } else if (k <= 1024) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, true, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, true, 2048);
#endif
        }
    } else {
        if (k == 1) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, false, 1);
        } else if (k <= 32) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, false, 32);
        } else if (k <= 64) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, false, 64);
        } else if (k <= 128) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, false, 128);
        } else if (k <= 256) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, false, 256);
        } else if (k <= 512) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, false, 512);
        } else if (k <= 1024) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, false, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BLOCK_SELECT_PAIR_CALL(float, int64_t, false, 2048);
#endif
        }
    }
}

}  // namespace core
}  // namespace cloudViewer
