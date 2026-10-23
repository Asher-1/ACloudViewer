#pragma once

#include <cstdint>
#include <utility>

namespace at {

struct PhiloxCudaState {
    std::uint64_t seed_ = 0;
    std::uint64_t offset_ = 0;
    bool captured_ = false;

    constexpr PhiloxCudaState(std::uint64_t seed, std::uint64_t offset)
        : seed_(seed), offset_(offset) {}
};

namespace cuda {
namespace philox {

__host__ __device__ inline std::pair<std::uint64_t, std::uint64_t> unpack(
        const PhiloxCudaState& state) {
    return {state.seed_, state.offset_};
}

}  // namespace philox
}  // namespace cuda
}  // namespace at
