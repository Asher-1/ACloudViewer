#pragma once

#include <ggml-backend.h>

#include <cctype>
#include <string>

namespace aicore::common {

/** Detect CUDA backend without linking libggml-cuda (dynamic backend mode). */
inline bool ggml_backend_is_cuda(ggml_backend_t backend) {
  if (backend == nullptr) {
    return false;
  }
  ggml_backend_dev_t dev = ggml_backend_get_device(backend);
  if (dev == nullptr) {
    return false;
  }
  if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
    return false;
  }
  const char *registry =
      ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
  if (registry == nullptr) {
    return false;
  }
  std::string name(registry);
  for (char &c : name) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return name == "cuda";
}

} // namespace aicore::common
