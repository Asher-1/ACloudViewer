#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sam3d {

// Reproduces torch.randn(..., device="cuda", dtype=torch.float32) from the
// PyTorch 2.5.1 CUDA distribution kernel.  The state is intentionally local
// to one image-to-3D session so unrelated CUDA work cannot perturb sampling.
class PytorchCudaNormalRng {
public:
    // A non-zero override freezes the PyTorch distribution-kernel launch
    // capacity so another backend can reproduce its state/scatter layout.
    explicit PytorchCudaNormalRng(uint64_t seed, uint32_t distribution_blocks = 0);

    PytorchCudaNormalRng(const PytorchCudaNormalRng&) = delete;
    PytorchCudaNormalRng& operator=(const PytorchCudaNormalRng&) = delete;

    static bool distribution_blocks(uint32_t& blocks, std::string& error);
    bool fill(std::vector<float>& values, std::string& error);

private:
    uint64_t seed_;
    uint32_t distribution_blocks_;
    uint64_t offset_ = 0;
};

}  // namespace sam3d
