#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "swin_backbone.hpp"

namespace rmbg {

class RmbgDeviceGraph {
public:
    RmbgDeviceGraph();
    ~RmbgDeviceGraph();
    RmbgDeviceGraph(const RmbgDeviceGraph &) = delete;
    RmbgDeviceGraph & operator=(const RmbgDeviceGraph &) = delete;

    bool init(ggml_backend_t backend, const WeightMap & weights, int input_size,
              std::string & err);
    bool forward(const std::vector<float> & input_nchw, std::vector<float> & alpha,
                 std::string & err);

    // Encoder taps are exposed only for numerical validation of the graph path.
    bool forward_encoder(const std::vector<float> & input_nchw,
                         std::vector<float> & x1, std::vector<float> & x2,
                         std::vector<float> & x3, std::vector<float> & x4,
                         std::string & err);
    bool forward_swin_debug(const std::vector<float> & input_nchw,
                            std::vector<float> & patch_tokens,
                            std::vector<float> & block0_tokens,
                            std::vector<float> & stage0_tokens,
                            std::vector<float> & stage1_tokens,
                            std::vector<float> & stage2_tokens,
                            std::vector<float> & stage3_tokens,
                            std::string & err);
    bool forward_block0_debug(const std::vector<float> & input_nchw,
                              std::vector<float> & patch_pre_norm,
                              std::vector<float> & norm1,
                              std::vector<float> & window0,
                              std::vector<float> & attended_window0,
                              std::vector<float> & after_attention,
                              std::string & err);

    size_t compute_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rmbg
