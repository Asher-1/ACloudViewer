#pragma once
#include <memory>
#include <string>
#include <vector>

#include "aicore/tagging_capi.h"
#include "common/image_utils.hpp"

namespace aicore {
namespace tagging {
struct Output {
    std::vector<float> logits;
    std::string caption;
    aicore_pipeline_timings timings{};
};
class Session {
public:
    explicit Session(const aicore_tagging_options& options);
    ~Session();
    Output infer(const img::Image& image,
                 const aicore_tagging_request& request,
                 uint64_t resize_mode = AICORE_TAGGING_RESIZE_OFFICIAL);
    const std::vector<std::string>& names() const;
    const std::vector<float>& thresholds() const;
    float selectionThreshold(size_t index,
                             const aicore_tagging_request& request) const;
    bool excluded(size_t index) const;
    const std::string& device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::vector<float> preprocess(const img::Image& image, int threads,
                              uint64_t resize_mode = AICORE_TAGGING_RESIZE_OFFICIAL);
}  // namespace tagging
}  // namespace aicore
