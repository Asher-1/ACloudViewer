#include "relateanything/options.hpp"
#include <limits>
#include <stdexcept>

namespace relateanything {
void ModelOptions::validate_model() const {
    if (backend != "cpu" && backend != "cuda" && backend != "vulkan") {
        throw std::invalid_argument("--backend must be cpu, cuda or vulkan");
    }
    if (quant_f32_input && backend != "cuda")
        throw std::invalid_argument("--quant-f32-input requires --backend cuda");
    if (!max_boxes || max_boxes > 10000 || !topk) throw std::invalid_argument("invalid model object or output limits");
    if (threads > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::invalid_argument("thread count exceeds integer range");
    if (backend == "cpu" && device_index != 0) throw std::invalid_argument("CPU device index must be zero");
}

} // namespace relateanything
