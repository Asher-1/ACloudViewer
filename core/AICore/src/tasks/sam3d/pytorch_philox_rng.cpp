// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pytorch_philox_rng.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>

namespace sam3d {
namespace {

constexpr uint64_t kThreadsPerBlock = 256;
constexpr uint64_t kValuesPerState = 4;
constexpr uint32_t kPhiloxM0 = 0xD2511F53U;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57U;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9U;
constexpr uint32_t kPhiloxW1 = 0xBB67AE85U;
constexpr float kTwoPow32Inverse = 2.3283064e-10f;
constexpr float kTwoPow32InverseTwoPi = 2.3283064e-10f * 6.2831855f;

struct PhiloxCounter {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
};

struct PhiloxKey {
    uint32_t x;
    uint32_t y;
};

uint32_t multiply_low(uint32_t left, uint32_t right, uint32_t& high) {
    const uint64_t product = static_cast<uint64_t>(left) * right;
    high = static_cast<uint32_t>(product >> 32U);
    return static_cast<uint32_t>(product);
}

PhiloxCounter philox_round(PhiloxCounter counter, PhiloxKey key) {
    uint32_t high0 = 0;
    uint32_t high1 = 0;
    const uint32_t low0 = multiply_low(kPhiloxM0, counter.x, high0);
    const uint32_t low1 = multiply_low(kPhiloxM1, counter.z, high1);
    return {high1 ^ counter.y ^ key.x, low1, high0 ^ counter.w ^ key.y, low0};
}

std::array<uint32_t, 4> philox4x32_10(PhiloxCounter counter, PhiloxKey key) {
    for (int round = 0; round < 10; ++round) {
        counter = philox_round(counter, key);
        if (round != 9) {
            key.x += kPhiloxW0;
            key.y += kPhiloxW1;
        }
    }
    return {counter.x, counter.y, counter.z, counter.w};
}

std::array<float, 4> box_muller4(const std::array<uint32_t, 4>& random_bits) {
    std::array<float, 4> samples{};
    for (size_t pair = 0; pair < 2; ++pair) {
        const float uniform_radius =
                static_cast<float>(random_bits[pair * 2]) * kTwoPow32Inverse +
                (kTwoPow32Inverse / 2.0f);
        const float uniform_angle =
                static_cast<float>(random_bits[pair * 2 + 1]) *
                        kTwoPow32InverseTwoPi +
                (kTwoPow32InverseTwoPi / 2.0f);
        const float radius = std::sqrt(-2.0f * std::log(uniform_radius));
        samples[pair * 2] = std::sin(uniform_angle) * radius;
        samples[pair * 2 + 1] = std::cos(uniform_angle) * radius;
    }
    return samples;
}

uint32_t grid_size_for(size_t count, uint32_t distribution_blocks) {
    const uint64_t requested =
            (static_cast<uint64_t>(count) + kThreadsPerBlock - 1U) /
            kThreadsPerBlock;
    return static_cast<uint32_t>(
            std::min<uint64_t>(requested, distribution_blocks));
}

bool normal_layout(size_t count,
                   uint32_t distribution_blocks,
                   uint32_t& grid_size,
                   uint64_t& stride,
                   uint64_t& iteration_count,
                   std::string& error) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        error = "PyTorch-compatible Philox RNG only supports up to INT_MAX "
                "elements per draw";
        return false;
    }
    grid_size = grid_size_for(count, distribution_blocks);
    if (grid_size == 0) {
        error = "PyTorch Philox distribution-block contract produced an empty "
                "launch grid";
        return false;
    }
    stride = static_cast<uint64_t>(grid_size) * kThreadsPerBlock;
    const uint64_t iteration_stride = stride * kValuesPerState;
    iteration_count = (static_cast<uint64_t>(count) + iteration_stride - 1U) /
                      iteration_stride;
    return true;
}

struct PhiloxWordStream {
    PhiloxWordStream(uint64_t seed, uint64_t sequence, uint64_t offset)
        : key{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32U)},
          counter{static_cast<uint32_t>((offset / 4U)),
                  static_cast<uint32_t>((offset / 4U) >> 32U),
                  static_cast<uint32_t>(sequence),
                  static_cast<uint32_t>(sequence >> 32U)},
          lane(static_cast<unsigned>(offset % 4U)) {}

    uint32_t next() {
        if (!ready) {
            words = philox4x32_10(counter, key);
            ready = true;
        }
        const uint32_t value = words[lane++];
        if (lane == words.size()) {
            lane = 0;
            ++counter.x;
            if (counter.x == 0) ++counter.y;
            ready = false;
        }
        return value;
    }

    PhiloxKey key;
    PhiloxCounter counter;
    std::array<uint32_t, 4> words{};
    unsigned lane = 0;
    bool ready = false;
};

int randperm_key_bits(size_t count) {
    const double n = static_cast<double>(count);
    const double log_threshold_12 = std::log(0.9) * 12.0;
    const double required = n - (6.0 * n * n + 1.0) / log_threshold_12;
    return std::min(64, static_cast<int>(std::ceil(std::log2(required))));
}

int64_t uint64_bits_as_int64(uint64_t value) {
    static_assert(sizeof(int64_t) == sizeof(uint64_t),
                  "unexpected integer width");
    int64_t signed_value = 0;
    std::memcpy(&signed_value, &value, sizeof(signed_value));
    return signed_value;
}

template <typename Key>
void shuffle_duplicate_key_islands(std::vector<Key>& keys,
                                   std::vector<uint32_t>& values,
                                   int bits,
                                   uint64_t seed,
                                   uint64_t offset) {
    const uint64_t mask =
            bits == 64 ? std::numeric_limits<uint64_t>::max()
                       : ((uint64_t{1} << static_cast<unsigned>(bits)) - 1U);
    const auto masked_key = [mask](Key key) {
        return static_cast<uint64_t>(key) & mask;
    };
    size_t begin = 0;
    while (begin < keys.size()) {
        size_t end = begin + 1;
        while (end < keys.size() &&
               masked_key(keys[begin]) == masked_key(keys[end]))
            ++end;
        if (end - begin > 1) {
            // PyTorch uses the sorted location as the Philox subsequence.
            PhiloxWordStream stream(seed, begin, offset);
            for (size_t index = end - begin - 1; index > 0; --index) {
                const size_t selected =
                        static_cast<size_t>(stream.next()) % (index + 1U);
                if (index != selected)
                    std::swap(values[begin + index], values[begin + selected]);
            }
        }
        begin = end;
    }
}

}  // namespace

PytorchPhiloxNormalRng::PytorchPhiloxNormalRng(uint64_t seed,
                                               uint32_t distribution_blocks)
    : seed_(seed), distribution_blocks_(distribution_blocks) {}

bool PytorchPhiloxNormalRng::fill(std::vector<float>& values,
                                  std::string& error) {
    if (values.empty()) return true;
    if (distribution_blocks_ == 0) {
        error = "PyTorch Philox distribution-block contract must be positive";
        return false;
    }
    uint32_t grid_size = 0;
    uint64_t stride = 0;
    uint64_t iteration_count = 0;
    if (!normal_layout(values.size(), distribution_blocks_, grid_size, stride,
                       iteration_count, error)) {
        return false;
    }
    const uint64_t iteration_stride = stride * kValuesPerState;
    const uint64_t count = values.size();
    const uint64_t offset_counter = offset_ / kValuesPerState;
    const PhiloxKey key{static_cast<uint32_t>(seed_),
                        static_cast<uint32_t>(seed_ >> 32U)};

    for (uint64_t thread_index = 0; thread_index < stride; ++thread_index) {
        for (uint64_t iteration = 0; iteration < iteration_count; ++iteration) {
            const uint64_t counter_index = offset_counter + iteration;
            const PhiloxCounter counter{
                    static_cast<uint32_t>(counter_index),
                    static_cast<uint32_t>(counter_index >> 32U),
                    static_cast<uint32_t>(thread_index),
                    static_cast<uint32_t>(thread_index >> 32U),
            };
            const std::array<float, 4> samples =
                    box_muller4(philox4x32_10(counter, key));
            const uint64_t output_base =
                    thread_index + iteration * iteration_stride;
            for (uint64_t lane = 0; lane < kValuesPerState; ++lane) {
                const uint64_t output_index = output_base + stride * lane;
                if (output_index < count)
                    values[static_cast<size_t>(output_index)] = samples[lane];
            }
        }
    }

    offset_ += iteration_count * kValuesPerState;
    return true;
}

bool PytorchPhiloxNormalRng::advance_normal(size_t count, std::string& error) {
    if (count == 0) return true;
    if (distribution_blocks_ == 0) {
        error = "PyTorch Philox distribution-block contract must be positive";
        return false;
    }
    uint32_t grid_size = 0;
    uint64_t stride = 0;
    uint64_t iteration_count = 0;
    if (!normal_layout(count, distribution_blocks_, grid_size, stride,
                       iteration_count, error)) {
        return false;
    }
    offset_ += iteration_count * kValuesPerState;
    return true;
}

bool PytorchPhiloxNormalRng::randperm(size_t count,
                                      std::vector<uint32_t>& values,
                                      std::string& error) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        error = "PyTorch-compatible randperm only supports up to INT_MAX "
                "elements";
        return false;
    }
    values.resize(count);
    std::iota(values.begin(), values.end(), 0U);
    if (count == 0) return true;
    if (distribution_blocks_ == 0) {
        error = "PyTorch Philox distribution-block contract must be positive";
        return false;
    }

    const int bits = randperm_key_bits(count);
    if (bits <= 0) {
        error = "PyTorch-compatible randperm produced an invalid key width";
        return false;
    }

    if (bits <= 32) {
        uint32_t grid_size = 0;
        uint64_t stride = 0;
        uint64_t iteration_count = 0;
        if (!normal_layout(count, distribution_blocks_, grid_size, stride,
                           iteration_count, error)) {
            return false;
        }
        const uint64_t iteration_stride = stride * kValuesPerState;
        const uint64_t offset_counter = offset_ / kValuesPerState;
        const PhiloxKey key{static_cast<uint32_t>(seed_),
                            static_cast<uint32_t>(seed_ >> 32U)};
        std::vector<int32_t> keys(count);
        constexpr uint64_t kRange =
                static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
        for (uint64_t thread_index = 0; thread_index < stride; ++thread_index) {
            for (uint64_t iteration = 0; iteration < iteration_count;
                 ++iteration) {
                const uint64_t counter_index = offset_counter + iteration;
                const PhiloxCounter counter{
                        static_cast<uint32_t>(counter_index),
                        static_cast<uint32_t>(counter_index >> 32U),
                        static_cast<uint32_t>(thread_index),
                        static_cast<uint32_t>(thread_index >> 32U)};
                const auto random_bits = philox4x32_10(counter, key);
                const uint64_t output_base =
                        thread_index + iteration * iteration_stride;
                for (uint64_t lane = 0; lane < kValuesPerState; ++lane) {
                    const uint64_t output_index = output_base + stride * lane;
                    if (output_index < count) {
                        keys[static_cast<size_t>(output_index)] = static_cast<
                                int32_t>(
                                static_cast<int64_t>(
                                        std::numeric_limits<int32_t>::min()) +
                                static_cast<int64_t>(random_bits[lane] %
                                                     kRange));
                    }
                }
            }
        }
        offset_ += iteration_count * kValuesPerState;
        std::vector<size_t> order(count);
        std::iota(order.begin(), order.end(), 0U);
        std::stable_sort(
                order.begin(), order.end(),
                [&keys, bits](size_t left, size_t right) {
                    const uint64_t mask =
                            bits == 32 ? std::numeric_limits<uint32_t>::max()
                                       : ((uint64_t{1}
                                           << static_cast<unsigned>(bits)) -
                                          1U);
                    return (static_cast<uint32_t>(keys[left]) & mask) <
                           (static_cast<uint32_t>(keys[right]) & mask);
                });
        std::vector<int32_t> sorted_keys(count);
        std::vector<uint32_t> sorted_values(count);
        for (size_t index = 0; index < count; ++index) {
            sorted_keys[index] = keys[order[index]];
            sorted_values[index] = values[order[index]];
        }
        shuffle_duplicate_key_islands(sorted_keys, sorted_values, bits, seed_,
                                      offset_);
        offset_ += count;
        values.swap(sorted_values);
        return true;
    }

    const uint32_t grid_size = grid_size_for(count, distribution_blocks_);
    if (grid_size == 0) {
        error = "PyTorch Philox distribution-block contract produced an empty "
                "launch grid";
        return false;
    }
    const uint64_t stride = static_cast<uint64_t>(grid_size) * kThreadsPerBlock;
    constexpr uint64_t kValuesPerState64 = 2;
    const uint64_t iteration_stride = stride * kValuesPerState64;
    const uint64_t iteration_count =
            (static_cast<uint64_t>(count) + iteration_stride - 1U) /
            iteration_stride;
    const uint64_t offset_counter = offset_ / kValuesPerState;
    const PhiloxKey key{static_cast<uint32_t>(seed_),
                        static_cast<uint32_t>(seed_ >> 32U)};
    std::vector<int64_t> keys(count);
    for (uint64_t thread_index = 0; thread_index < stride; ++thread_index) {
        for (uint64_t iteration = 0; iteration < iteration_count; ++iteration) {
            const uint64_t counter_index = offset_counter + iteration;
            const PhiloxCounter counter{
                    static_cast<uint32_t>(counter_index),
                    static_cast<uint32_t>(counter_index >> 32U),
                    static_cast<uint32_t>(thread_index),
                    static_cast<uint32_t>(thread_index >> 32U)};
            const auto random_bits = philox4x32_10(counter, key);
            const uint64_t output_base =
                    thread_index + iteration * iteration_stride;
            for (uint64_t lane = 0; lane < kValuesPerState64; ++lane) {
                const uint64_t output_index = output_base + stride * lane;
                if (output_index < count) {
                    const uint64_t random64 =
                            (static_cast<uint64_t>(random_bits[lane * 2U])
                             << 32U) |
                            random_bits[lane * 2U + 1U];
                    const uint64_t mapped =
                            random64 % std::numeric_limits<uint64_t>::max();
                    keys[static_cast<size_t>(output_index)] =
                            uint64_bits_as_int64(mapped ^ (uint64_t{1} << 63U));
                }
            }
        }
    }
    offset_ += iteration_count * kValuesPerState;
    std::vector<size_t> order(count);
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(),
                     [&keys, bits](size_t left, size_t right) {
                         const uint64_t mask =
                                 bits == 64
                                         ? std::numeric_limits<uint64_t>::max()
                                         : ((uint64_t{1}
                                             << static_cast<unsigned>(bits)) -
                                            1U);
                         return (static_cast<uint64_t>(keys[left]) & mask) <
                                (static_cast<uint64_t>(keys[right]) & mask);
                     });
    std::vector<int64_t> sorted_keys(count);
    std::vector<uint32_t> sorted_values(count);
    for (size_t index = 0; index < count; ++index) {
        sorted_keys[index] = keys[order[index]];
        sorted_values[index] = values[order[index]];
    }
    shuffle_duplicate_key_islands(sorted_keys, sorted_values, bits, seed_,
                                  offset_);
    offset_ += count;
    values.swap(sorted_values);
    return true;
}

}  // namespace sam3d
