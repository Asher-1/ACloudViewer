// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cuda.h>

#include "core/nns/kernel/Limits.cuh"
#include "core/nns/kernel/Pair.cuh"

namespace cloudViewer {
namespace core {

template <typename T>
struct Sum {
    __device__ inline T operator()(T a, T b) const { return a + b; }

    inline __device__ T identity() const { return 0.0; }
};

template <typename T>
struct Min {
    __device__ inline T operator()(T a, T b) const { return a < b ? a : b; }

    inline __device__ T identity() const { return Limits<T>::getMax(); }
};

template <typename T>
struct Max {
    __device__ inline T operator()(T a, T b) const { return a > b ? a : b; }

    inline __device__ T identity() const { return Limits<T>::getMin(); }
};

/// Used for producing segmented prefix scans; the value of the Pair
/// denotes the start of a new segment for the scan
template <typename T, typename ReduceOp>
struct SegmentedReduce {
    inline __device__ SegmentedReduce(const ReduceOp& o) : op(o) {}

    __device__ inline Pair<T, bool> operator()(const Pair<T, bool>& a,
                                               const Pair<T, bool>& b) const {
        return Pair<T, bool>(b.v ? b.k : op(a.k, b.k), a.v || b.v);
    }

    inline __device__ Pair<T, bool> identity() const {
        return Pair<T, bool>(op.identity(), false);
    }

    ReduceOp op;
};

}  // namespace core
}  // namespace cloudViewer
