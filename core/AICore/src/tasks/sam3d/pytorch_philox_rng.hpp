#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sam3d {

// Host implementation of the Philox4x32-10 state and scatter layout used by
// the verified CUDA torch.randn compatibility kernel.  The number of
// distribution blocks is part of the PyTorch CUDA execution contract: it is
// derived from the reference device's launch capacity, not from the GGML
// backend.  Callers must pass the value recorded for the PyTorch oracle.
class PytorchPhiloxNormalRng {
public:
    PytorchPhiloxNormalRng(uint64_t seed, uint32_t distribution_blocks);

    PytorchPhiloxNormalRng(const PytorchPhiloxNormalRng&) = delete;
    PytorchPhiloxNormalRng& operator=(const PytorchPhiloxNormalRng&) = delete;

    bool fill(std::vector<float>& values, std::string& error);
    // Advance the same CUDA distribution-kernel state without materializing
    // values. CUDA sessions use this to keep a portable randperm oracle in
    // lockstep with the device normal sampler.
    bool advance_normal(size_t count, std::string& error);

    // Reproduces the PyTorch 2.5.1 CUDA randperm path: generate random keys
    // from the current Philox state, radix-sort them, then shuffle key
    // collisions with the separately reserved Philox range. Values are the
    // first `count` entries of torch.randperm(count, device="cuda").
    bool randperm(size_t count, std::vector<uint32_t>& values, std::string& error);

private:
    uint64_t seed_;
    uint32_t distribution_blocks_;
    uint64_t offset_ = 0;
};

// Exact host counterpart of
// sam3d_objects.pipeline.inference_utils.downsample_sparse_structure().
// `rng` must already be advanced through every normal draw preceding the
// coordinate selection in the official pipeline.
std::vector<int32_t> downsample_sparse_coords_pytorch(
    const std::vector<int32_t>& coordinates,
    PytorchPhiloxNormalRng& rng,
    std::string& error,
    bool& randomly_subsampled,
    int64_t max_coordinates = 42000,
    int downsample_factor = 2);

}  // namespace sam3d
