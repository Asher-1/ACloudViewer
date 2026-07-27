#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace matchanything {

enum class Variant { kEloftr = 0, kRoma = 1 };

struct MatchPair {
  float x0 = 0.0f;
  float y0 = 0.0f;
  float x1 = 0.0f;
  float y1 = 0.0f;
  float score = 0.0f;
};

struct MatchAnythingOptions {
  std::string model_path;
  std::string device = "cpu"; // cpu | cuda | vulkan
  int32_t num_threads = 4;
  Variant variant = Variant::kEloftr;
};

struct MatchAnythingResult {
  std::vector<MatchPair> matches;
};

class MatchAnythingMatcher {
public:
  virtual ~MatchAnythingMatcher() = default;
  virtual bool MatchGray(const uint8_t *img0, const uint8_t *img1, int32_t w,
                         int32_t h, int32_t stride, MatchAnythingResult *result) = 0;
  virtual const std::string &Device() const = 0;
  virtual const std::string &VariantName() const = 0;
  virtual const std::string &Error() const = 0;
};

std::unique_ptr<MatchAnythingMatcher> CreateMatchAnythingMatcher(
    const MatchAnythingOptions &options, std::string *error = nullptr);

} // namespace matchanything
