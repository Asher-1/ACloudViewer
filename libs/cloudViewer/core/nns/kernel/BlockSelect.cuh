// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "core/Tensor.h"
#include "core/nns/kernel/Select.cuh"

namespace cloudViewer {
namespace core {

template <typename K,
          typename IndexType,
          bool Dir,
          int NumWarpQ,
          int NumThreadQ,
          int ThreadsPerBlock>
__global__ void blockSelect(K* in,
                            K* outK,
                            IndexType* outV,
                            K initK,
                            IndexType initV,
                            int k,
                            int dim,
                            int num_points) {
    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

    __shared__ K smemK[kNumWarps * NumWarpQ];
    __shared__ IndexType smemV[kNumWarps * NumWarpQ];

    BlockSelect<K, IndexType, Dir, NumWarpQ, NumThreadQ, ThreadsPerBlock> heap(
            initK, initV, smemK, smemV, k);

    // Grid is exactly sized to rows available
    int row = blockIdx.x;

    int i = threadIdx.x;
    K* inStart = in + dim * row + i;

    // Whole warps must participate in the selection
    int limit = (dim / kWarpSize) * kWarpSize;

    for (; i < limit; i += ThreadsPerBlock) {
        heap.add(*inStart, (IndexType)i);
        inStart += ThreadsPerBlock;
    }

    // Handle last remainder fraction of a warp of elements
    if (i < dim) {
        heap.addThreadQ(*inStart, (IndexType)i);
    }

    heap.reduce();

    for (int i = threadIdx.x; i < k; i += ThreadsPerBlock) {
        *(outK + row * dim + i) = smemK[i];
        *(outV + row * dim + i) = smemV[i];
    }
}

template <typename K,
          typename IndexType,
          bool Dir,
          int NumWarpQ,
          int NumThreadQ,
          int ThreadsPerBlock>
__global__ void blockSelectPair(K* inK,
                                IndexType* inV,
                                K* outK,
                                IndexType* outV,
                                K initK,
                                IndexType initV,
                                int k,
                                int dim,
                                int num_points) {
    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

    __shared__ K smemK[kNumWarps * NumWarpQ];
    __shared__ IndexType smemV[kNumWarps * NumWarpQ];

    BlockSelect<K, IndexType, Dir, NumWarpQ, NumThreadQ, ThreadsPerBlock> heap(
            initK, initV, smemK, smemV, k);

    // Grid is exactly sized to rows available
    int row = blockIdx.x;

    int i = threadIdx.x;
    K* inKStart = &inK[row * dim + i];
    IndexType* inVStart = &inV[row * dim + i];

    // Whole warps must participate in the selection
    int limit = (dim / kWarpSize) * kWarpSize;

    for (; i < limit; i += ThreadsPerBlock) {
        heap.add(*inKStart, *inVStart);
        inKStart += ThreadsPerBlock;
        inVStart += ThreadsPerBlock;
    }

    // Handle last remainder fraction of a warp of elements
    if (i < dim) {
        heap.addThreadQ(*inKStart, *inVStart);
    }

    heap.reduce();

    for (int i = threadIdx.x; i < k; i += ThreadsPerBlock) {
        outK[row * k + i] = smemK[i];
        outV[row * k + i] = smemV[i];
    }
}

void runBlockSelectPair(cudaStream_t stream,
                        float* inK,
                        int32_t* inV,
                        float* outK,
                        int32_t* outV,
                        bool dir,
                        int k,
                        int dim,
                        int num_points);

void runBlockSelectPair(cudaStream_t stream,
                        float* inK,
                        int64_t* inV,
                        float* outK,
                        int64_t* outV,
                        bool dir,
                        int k,
                        int dim,
                        int num_points);

void runBlockSelectPair(cudaStream_t stream,
                        double* inK,
                        int32_t* inV,
                        double* outK,
                        int32_t* outV,
                        bool dir,
                        int k,
                        int dim,
                        int num_points);

void runBlockSelectPair(cudaStream_t stream,
                        double* inK,
                        int64_t* inV,
                        double* outK,
                        int64_t* outV,
                        bool dir,
                        int k,
                        int dim,
                        int num_points);

}  // namespace core
}  // namespace cloudViewer
