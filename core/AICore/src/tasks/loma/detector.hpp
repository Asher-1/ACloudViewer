// DaD keypoint detector used by COLMAP LoMa.  The model is lowered to ggml;
// this interface deliberately contains no ONNX types or runtime dependency.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "aicore/loma_capi.h"

namespace aicore {
namespace loma {

struct DetectorOptions {
    // DaD's first ggml lowering is numerically gated on the CPU backend.
    std::string device = "cpu";
    int32_t num_threads = 0;
    int32_t max_keypoints = 2048;
};

struct DetectorResult {
    std::vector<aicore_loma_keypoint> keypoints;
    std::vector<float> scores;
};

class Detector {
public:
    Detector();
    ~Detector();

    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    bool Load(const std::string& gguf_path, const DetectorOptions& options);
    bool Detect(const aicore_loma_rgb_image& image, DetectorResult* result);
    const std::string& error() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace loma
}  // namespace aicore
