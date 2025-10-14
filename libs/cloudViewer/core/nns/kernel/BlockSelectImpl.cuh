// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cloudViewer/core/CUDAUtils.h"
#include "core/nns/kernel/BlockSelect.cuh"
#include "core/nns/kernel/Limits.cuh"

#define BLOCK_SELECT_IMPL(TYPE, TINDEX, DIR, WARP_Q, THREAD_Q)                 \
    void runBlockSelect_##TYPE##_##TINDEX##_##DIR##_##WARP_Q##_(               \
            cudaStream_t stream, TYPE* in, TYPE* outK, TINDEX* outV, bool dir, \
            int k, int dim, int num_points) {                                  \
        auto grid = dim3(num_points);                                          \
                                                                               \
        constexpr int kBlockSelectNumThreads =                                 \
                sizeof(TYPE) == 4 ? ((WARP_Q <= 1024) ? 128 : 64)              \
                                  : ((WARP_Q <= 512) ? 64 : 32);               \
        auto block = dim3(kBlockSelectNumThreads);                             \
                                                                               \
        CLOUDVIEWER_ASSERT(k <= WARP_Q);                                       \
        CLOUDVIEWER_ASSERT(dir == DIR);                                        \
                                                                               \
        auto kInit = dir ? Limits<TYPE>::getMin() : Limits<TYPE>::getMax();    \
        auto vInit = -1;                                                       \
                                                                               \
        blockSelect<TYPE, TINDEX, DIR, WARP_Q, THREAD_Q,                       \
                    kBlockSelectNumThreads><<<grid, block, 0, stream>>>(       \
                in, outK, outV, kInit, vInit, k, dim, num_points);             \
    }                                                                          \
                                                                               \
    void runBlockSelectPair_##TYPE##_##TINDEX##_##DIR##_##WARP_Q##_(           \
            cudaStream_t stream, TYPE* inK, TINDEX* inV, TYPE* outK,           \
            TINDEX* outV, bool dir, int k, int dim, int num_points) {          \
        auto grid = dim3(num_points);                                          \
                                                                               \
        constexpr int kBlockSelectNumThreads =                                 \
                sizeof(TYPE) == 4 ? ((WARP_Q <= 1024) ? 128 : 64)              \
                                  : ((WARP_Q <= 512) ? 64 : 32);               \
        auto block = dim3(kBlockSelectNumThreads);                             \
                                                                               \
        CLOUDVIEWER_ASSERT(k <= WARP_Q);                                       \
        CLOUDVIEWER_ASSERT(dir == DIR);                                        \
                                                                               \
        auto kInit = dir ? Limits<TYPE>::getMin() : Limits<TYPE>::getMax();    \
        auto vInit = -1;                                                       \
                                                                               \
        blockSelectPair<TYPE, TINDEX, DIR, WARP_Q, THREAD_Q,                   \
                        kBlockSelectNumThreads><<<grid, block, 0, stream>>>(   \
                inK, inV, outK, outV, kInit, vInit, k, dim, num_points);       \
    }

#define BLOCK_SELECT_CALL(TYPE, TINDEX, DIR, WARP_Q)        \
    runBlockSelect_##TYPE##_##TINDEX##_##DIR##_##WARP_Q##_( \
            stream, in, outK, outV, dir, k, dim, num_points)

#define BLOCK_SELECT_PAIR_CALL(TYPE, TINDEX, DIR, WARP_Q)       \
    runBlockSelectPair_##TYPE##_##TINDEX##_##DIR##_##WARP_Q##_( \
            stream, inK, inV, outK, outV, dir, k, dim, num_points)
