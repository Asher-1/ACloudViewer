#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "aicore/loma_capi.h"

namespace aicore::loma {

struct DescriptorOptions {
    std::string device = "cpu";
    int32_t num_threads = 0;
    // These controls are private to the white-box DINOv2 numerical gate and
    // are never exposed through the public C API.
    bool allow_ungated_g_for_validation = false;
    bool trace_g_tokens_for_validation = false;
    // -3 traces patch embedding, -2 includes positional encoding, and [0, 23]
    // traces a transformer block output. -1 keeps the final token trace.
    int32_t trace_g_block_for_validation = -1;
};

class Descriptor {
public:
    Descriptor();
    ~Descriptor();
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    bool Load(const std::string& gguf_path, const DescriptorOptions& options);
    bool Describe(const aicore_loma_rgb_image& image,
                  const aicore_loma_keypoint* keypoints,
                  int32_t count,
                  int32_t keypoint_image_width,
                  int32_t keypoint_image_height,
                  std::vector<float>* descriptors);
    int32_t descriptor_dim() const;
    const std::string& error() const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aicore::loma
