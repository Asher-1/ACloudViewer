// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <mutex>
#include <random>
#include <vector>

namespace cloudViewer {
namespace utility {
namespace random {

inline std::mt19937* GetEngine() {
    thread_local static std::mt19937 engine{std::random_device{}()};
    return &engine;
}

inline std::mutex* GetMutex() {
    static std::mutex mtx;
    return &mtx;
}

template <typename T>
class UniformRealGenerator {
public:
    UniformRealGenerator(T a, T b) : distribution_(a, b) {}

    T operator()() {
        std::lock_guard<std::mutex> lock(*GetMutex());
        return distribution_(*GetEngine());
    }

private:
    std::uniform_real_distribution<T> distribution_;
};

template <typename IndexType = size_t>
class DiscreteGenerator {
public:
    template <typename WeightType>
    DiscreteGenerator(const WeightType* weights, size_t count) {
        cumulative_weights_.resize(count);
        double running_sum = 0.0;
        for (size_t i = 0; i < count; ++i) {
            running_sum += static_cast<double>(weights[i]);
            cumulative_weights_[i] = running_sum;
        }
        total_weight_ = running_sum > 0.0 ? running_sum : 1.0;
    }

    template <typename WeightType>
    DiscreteGenerator(const WeightType* begin, const WeightType* end) {
        const size_t count = static_cast<size_t>(end - begin);
        cumulative_weights_.resize(count);
        double running_sum = 0.0;
        for (size_t i = 0; i < count; ++i) {
            running_sum += static_cast<double>(begin[i]);
            cumulative_weights_[i] = running_sum;
        }
        total_weight_ = running_sum > 0.0 ? running_sum : 1.0;
    }

    IndexType operator()() {
        std::lock_guard<std::mutex> lock(*GetMutex());
        std::uniform_real_distribution<double> dist(0.0, total_weight_);
        const double r = dist(*GetEngine());
        auto it = std::lower_bound(cumulative_weights_.begin(),
                                   cumulative_weights_.end(), r);
        if (it == cumulative_weights_.end()) {
            return static_cast<IndexType>(cumulative_weights_.empty()
                                                  ? 0
                                                  : cumulative_weights_.size() -
                                                            1);
        }
        return static_cast<IndexType>(it - cumulative_weights_.begin());
    }

private:
    std::vector<double> cumulative_weights_;
    double total_weight_ = 1.0;
};

}  // namespace random
}  // namespace utility
}  // namespace cloudViewer

namespace cloudViewer {
namespace utility {
namespace random {

template <typename T>
class NormalGenerator {
public:
    NormalGenerator(T mean, T stddev) : distribution_(mean, stddev) {}

    T operator()() {
        std::lock_guard<std::mutex> lock(*GetMutex());
        return distribution_(*GetEngine());
    }

private:
    std::normal_distribution<T> distribution_;
};

}  // namespace random
}  // namespace utility
}  // namespace cloudViewer


