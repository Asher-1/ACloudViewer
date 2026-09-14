// LoMa GGUF quantization. Output remains a standard GGUF file and is loaded by
// the production DaD/DeDoDe/LoMa graph; ONNX is never needed at runtime.
#pragma once

#include <string>

namespace aicore::loma {

bool QuantizeModel(const std::string& input_gguf,
                   const std::string& output_gguf,
                   const std::string& type,
                   std::string* error = nullptr);

}  // namespace aicore::loma
