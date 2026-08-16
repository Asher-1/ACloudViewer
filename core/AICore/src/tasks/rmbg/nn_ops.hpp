#pragma once

#include "swin_backbone.hpp"
#include <string>
#include <vector>

namespace rmbg {

struct BatchNorm2dParams {
    std::vector<float> weight;
    std::vector<float> bias;
    std::vector<float> running_mean;
    std::vector<float> running_var;
    float eps = 1e-5f;
};

struct Conv2dParams {
    std::vector<float> weight; // [OC, IC, KH, KW]
    std::vector<float> bias;
    int ic = 0, oc = 0, kh = 1, kw = 1;
};

struct DeformConv2dParams {
    Conv2dParams offset;
    Conv2dParams modulator;
    Conv2dParams regular;
    int padding = 0;
};

void conv2d_nchw(const std::vector<float> & in, int N, int C, int H, int W,
                 const Conv2dParams & p, int stride, int pad,
                 std::vector<float> & out);

void batch_norm2d_nchw(const std::vector<float> & in, int N, int C, int H, int W,
                       const BatchNorm2dParams & p, std::vector<float> & out);

void relu_inplace(std::vector<float> & x);
void sigmoid_inplace(std::vector<float> & x);

void bilinear_resize_nchw(const std::vector<float> & in, int N, int C, int H, int W,
                          int OH, int OW, std::vector<float> & out);

/// Concatenate two NCHW tensors along channel dim (same H,W).
void concat_nchw_channel(const std::vector<float> & a, int Ca,
                         const std::vector<float> & b, int Cb,
                         int H, int W, std::vector<float> & out);

void deform_conv2d_nchw(const std::vector<float> & in, int N, int C, int H, int W,
                        const DeformConv2dParams & p,
                        std::vector<float> & out);

bool load_conv2d(const WeightMap & w, const std::string & prefix, Conv2dParams & out,
                 std::string & err);
bool load_batch_norm2d(const WeightMap & w, const std::string & prefix, BatchNorm2dParams & out,
                       std::string & err);
bool load_deform_conv2d(const WeightMap & w, const std::string & prefix, DeformConv2dParams & out,
                        std::string & err);

bool aspp_deform_branch_forward(const std::vector<float> & in, int N, int C, int H, int W,
                                const WeightMap & w, const std::string & prefix,
                                std::vector<float> & out, std::string & err);

bool aspp_deformable_forward(const std::vector<float> & in, int N, int C, int H, int W,
                             const WeightMap & w, const std::string & prefix,
                             std::vector<float> & out, std::string & err);

bool basic_dec_blk_forward(const std::vector<float> & in, int N, int C, int H, int W,
                           const WeightMap & w, const std::string & prefix,
                           std::vector<float> & out, std::string & err);

bool lateral_block_forward(const std::vector<float> & in, int N, int C, int H, int W,
                           const WeightMap & w, const std::string & prefix,
                           std::vector<float> & out, std::string & err);

bool simple_convs_forward(const std::vector<float> & in, int N, int C, int H, int W,
                            const WeightMap & w, const std::string & prefix,
                            std::vector<float> & out, std::string & err);

void image2patches_split_nchw(const std::vector<float> & image, int H, int W,
                              int patch_h, int patch_w,
                              std::vector<float> & patches);

void add_nchw(const std::vector<float> & a, const std::vector<float> & b,
              int C, int H, int W, std::vector<float> & out);

bool gdt_attn_forward(const std::vector<float> & in, int N, int C, int H, int W,
                      const WeightMap & w, const std::string & gdt_prefix,
                      const std::string & attn_prefix,
                      std::vector<float> & out, std::string & err);

bool squeeze_module_forward(const std::vector<float> & in, int N, int C, int H, int W,
                            const WeightMap & w,
                            std::vector<float> & out, std::string & err);

} // namespace rmbg
