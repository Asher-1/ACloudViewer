// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cuda.h>

#include "core/nns/kernel/PtxUtils.cuh"
#include "core/nns/kernel/WarpShuffle.cuh"

namespace cloudViewer {
namespace core {

/// A simple pair type for CUDA device usage
template <typename K, typename V>
struct Pair {
    constexpr __device__ inline Pair() {}

    constexpr __device__ inline Pair(K key, V value) : k(key), v(value) {}

    __device__ inline bool operator==(const Pair<K, V>& rhs) const {
        return k == rhs.k && v == rhs.v;
    }

    __device__ inline bool operator!=(const Pair<K, V>& rhs) const {
        return !operator==(rhs);
    }

    __device__ inline bool operator<(const Pair<K, V>& rhs) const {
        return k < rhs.k || (k == rhs.k && v < rhs.v);
    }

    __device__ inline bool operator>(const Pair<K, V>& rhs) const {
        return k > rhs.k || (k == rhs.k && v > rhs.v);
    }

    K k;
    V v;
};

template <typename T, typename U>
inline __device__ Pair<T, U> shfl_up(const Pair<T, U>& pair,
                                     unsigned int delta,
                                     int width = kWarpSize) {
    return Pair<T, U>(shfl_up(pair.k, delta, width),
                      shfl_up(pair.v, delta, width));
}

template <typename T, typename U>
inline __device__ Pair<T, U> shfl_xor(const Pair<T, U>& pair,
                                      int laneMask,
                                      int width = kWarpSize) {
    return Pair<T, U>(shfl_xor(pair.k, laneMask, width),
                      shfl_xor(pair.v, laneMask, width));
}

}  // namespace core
}  // namespace cloudViewer
