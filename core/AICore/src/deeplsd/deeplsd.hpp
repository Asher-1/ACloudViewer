#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace deeplsd {

struct LineSegment {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
};

struct DeepLSDOptions {
  std::string model_path;
  std::string device = "cpu"; // cpu | cuda | vulkan
  int32_t num_threads = 4;
  bool use_ggml_cnn = true;
};

struct DeepLSDResult {
  std::vector<float> distance_field;
  std::vector<float> angle_field;
  int32_t width = 0;
  int32_t height = 0;
  std::vector<LineSegment> segments;
};

class DeepLSDExtractor {
public:
  virtual ~DeepLSDExtractor() = default;
  virtual bool ExtractFromGray(const uint8_t *gray, int32_t width, int32_t height,
                               int32_t row_stride, DeepLSDResult *result) = 0;
  virtual const std::string &Device() const = 0;
  virtual const std::string &Error() const = 0;
};

std::unique_ptr<DeepLSDExtractor> CreateDeepLSDExtractor(const DeepLSDOptions &options,
                                                         std::string *error = nullptr);

} // namespace deeplsd
