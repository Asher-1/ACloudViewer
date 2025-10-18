// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <chrono>
#include <memory>
#include <random>
#include <thread>

#include "util/logging.h"

namespace colmap {

extern thread_local std::unique_ptr<std::mt19937> PRNG;

static int kDefaultPRNGSeed = 0;

// Initialize the PRNG with the given seed.
//
// @param seed   The seed for the PRNG. If the seed is -1, the current time
//               is used as the seed.
void SetPRNGSeed(unsigned seed = kDefaultPRNGSeed);

// Generate uniformly distributed random integer number.
//
// This implementation is unbiased and thread-safe in contrast to `rand()`.
template <typename T>
T RandomInteger(const T min, const T max);

// Generate uniformly distributed random real number.
//
// This implementation is unbiased and thread-safe in contrast to `rand()`.
template <typename T>
T RandomReal(const T min, const T max);

// Generate Gaussian distributed random real number.
//
// This implementation is unbiased and thread-safe in contrast to `rand()`.
template <typename T>
T RandomGaussian(const T mean, const T stddev);

// Fisher-Yates shuffling.
//
// Note that the vector may not contain more values than UINT32_MAX. This
// restriction comes from the fact that the 32-bit version of the
// Mersenne Twister PRNG is significantly faster.
//
// @param elems            Vector of elements to shuffle.
// @param num_to_shuffle   Optional parameter, specifying the number of first
//                         N elements in the vector to shuffle.
template <typename T>
void Shuffle(const uint32_t num_to_shuffle, std::vector<T>* elems);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
T RandomInteger(const T min, const T max) {
    if (PRNG == nullptr) {
        SetPRNGSeed();
    }

    std::uniform_int_distribution<T> distribution(min, max);

    return distribution(*PRNG);
}

template <typename T>
T RandomReal(const T min, const T max) {
    if (PRNG == nullptr) {
        SetPRNGSeed();
    }

    std::uniform_real_distribution<T> distribution(min, max);

    return distribution(*PRNG);
}

template <typename T>
T RandomGaussian(const T mean, const T stddev) {
    if (PRNG == nullptr) {
        SetPRNGSeed();
    }

    std::normal_distribution<T> distribution(mean, stddev);
    return distribution(*PRNG);
}

template <typename T>
void Shuffle(const uint32_t num_to_shuffle, std::vector<T>* elems) {
    CHECK_LE(num_to_shuffle, elems->size());
    const uint32_t last_idx = static_cast<uint32_t>(elems->size() - 1);
    for (uint32_t i = 0; i < num_to_shuffle; ++i) {
        const auto j = RandomInteger<uint32_t>(i, last_idx);
        std::swap((*elems)[i], (*elems)[j]);
    }
}

}  // namespace colmap
