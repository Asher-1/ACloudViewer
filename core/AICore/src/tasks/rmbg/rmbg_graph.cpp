#include "rmbg_graph.hpp"

#include "ggml-alloc.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <utility>

namespace rmbg {
namespace {

constexpr size_t kMaxGraphNodes = 65536;

struct StaticBlob {
    ggml_tensor * tensor = nullptr;
    std::vector<uint8_t> bytes;
};

struct SwinTaps {
    ggml_tensor * patch_pre_norm = nullptr;
    ggml_tensor * patch = nullptr;
    ggml_tensor * block0 = nullptr;
    ggml_tensor * block0_norm1 = nullptr;
    ggml_tensor * block0_windows = nullptr;
    ggml_tensor * block0_attended_windows = nullptr;
    ggml_tensor * block0_after_attention = nullptr;
    ggml_tensor * raw[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor * x1 = nullptr;
    ggml_tensor * x2 = nullptr;
    ggml_tensor * x3 = nullptr;
    ggml_tensor * x4 = nullptr;
};

struct EncoderTaps {
    ggml_tensor * x1 = nullptr;
    ggml_tensor * x2 = nullptr;
    ggml_tensor * x3 = nullptr;
    ggml_tensor * x4 = nullptr;
};

struct GraphBuilder {
    ggml_context * stat = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    bool use_cuda_custom = false;
    bool use_backend_custom = false;
    bool use_vulkan_custom = false;
    bool is_cpu_backend = false;
    const WeightMap & weights;
    std::unordered_map<std::string, ggml_tensor *> weight_cache;
    std::vector<StaticBlob> blobs;
    int constant_id = 0;

    GraphBuilder(ggml_context * stat_, ggml_context * ctx_, ggml_backend_t backend_,
                 const WeightMap & weights_)
        : stat(stat_), ctx(ctx_), backend(backend_), weights(weights_) {
        const char * name = ggml_backend_name(backend);
        use_cuda_custom = name && std::strstr(name, "CUDA");
        use_backend_custom = name && (std::strstr(name, "CUDA") || std::strstr(name, "Vulkan"));
        use_vulkan_custom = name && std::strstr(name, "Vulkan");
        is_cpu_backend = name && std::strstr(name, "CPU");
    }

    ggml_tensor * weight(const std::string & name) {
        auto found = weight_cache.find(name);
        if (found != weight_cache.end()) return found->second;
        const WeightTensor * source = weights.get_tensor(name.c_str());
        if (!source || source->shape.empty() || source->shape.size() > GGML_MAX_DIMS) return nullptr;
        const bool matrix_weight = source->shape.size() >= 2 &&
            name.size() >= 7 && name.compare(name.size() - 7, 7, "_weight") == 0;
        // CUDA's custom Swin node and the CPU backend's ggml_mul_mat both
        // require F32 weights.  Vulkan keeps native F16 matrices for its own
        // shaders.  Expanding a compact F16 model to F32 once during graph
        // initialization satisfies the CUDA and CPU constraints without
        // affecting Vulkan's fast path.
        const bool preserve_type = source->type == GGML_TYPE_F32 ||
            (matrix_weight && use_vulkan_custom);
        const std::vector<float> * f32 = preserve_type ? nullptr : weights.get_f32(name.c_str());
        if (!preserve_type && !f32) return nullptr;
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (size_t i = 0; i < source->shape.size(); ++i) ne[i] = source->shape[source->shape.size() - 1 - i];
        const enum ggml_type type = preserve_type ? source->type : GGML_TYPE_F32;
        ggml_tensor * tensor = ggml_new_tensor(stat, type, (int) source->shape.size(), ne);
        ggml_set_name(tensor, name.c_str());
        StaticBlob blob;
        blob.tensor = tensor;
        if (preserve_type) blob.bytes = source->bytes;
        else {
            blob.bytes.resize(f32->size() * sizeof(float));
            std::memcpy(blob.bytes.data(), f32->data(), blob.bytes.size());
        }
        blobs.push_back(std::move(blob));
        weight_cache.emplace(name, tensor);
        return tensor;
    }

    ggml_tensor * weight_f16(const std::string & name) {
        const std::string cache_name = name + "__f16";
        auto found = weight_cache.find(cache_name);
        if (found != weight_cache.end()) return found->second;
        const WeightTensor * source = weights.get_tensor(name.c_str());
        if (!source || source->shape.empty() || source->shape.size() > GGML_MAX_DIMS) return nullptr;
        if (source->type == GGML_TYPE_F16) return weight(name);
        const std::vector<float> * f32 = weights.get_f32(name.c_str());
        if (!f32) return nullptr;
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (size_t i = 0; i < source->shape.size(); ++i) {
            ne[i] = source->shape[source->shape.size() - 1 - i];
        }
        ggml_tensor * tensor = ggml_new_tensor(stat, GGML_TYPE_F16,
            (int) source->shape.size(), ne);
        ggml_set_name(tensor, cache_name.c_str());
        StaticBlob blob;
        blob.tensor = tensor;
        std::vector<ggml_fp16_t> converted(f32->size());
        for (size_t i = 0; i < f32->size(); ++i) converted[i] = ggml_fp32_to_fp16((*f32)[i]);
        blob.bytes.resize(converted.size() * sizeof(ggml_fp16_t));
        std::memcpy(blob.bytes.data(), converted.data(), blob.bytes.size());
        blobs.push_back(std::move(blob));
        weight_cache.emplace(cache_name, tensor);
        return tensor;
    }

    template <typename T>
    ggml_tensor * constant(enum ggml_type type, const std::vector<int64_t> & ne,
                           const std::vector<T> & data, const char * stem) {
        ggml_tensor * tensor = ggml_new_tensor(stat, type, (int) ne.size(), ne.data());
        const std::string name = std::string(stem) + "_" + std::to_string(constant_id++);
        ggml_set_name(tensor, name.c_str());
        StaticBlob blob;
        blob.tensor = tensor;
        blob.bytes.resize(data.size() * sizeof(T));
        std::memcpy(blob.bytes.data(), data.data(), blob.bytes.size());
        blobs.push_back(std::move(blob));
        return tensor;
    }

    ggml_tensor * zeros(int64_t ne0, int64_t ne1 = 1) {
        std::vector<float> data((size_t) ne0 * ne1, 0.f);
        return constant(GGML_TYPE_F32, {ne0, ne1}, data, "zero");
    }

    ggml_tensor * scalar(float value) {
        return constant(GGML_TYPE_F32, {1}, std::vector<float>{value}, "scalar");
    }

    ggml_tensor * add_bias_tokens(ggml_tensor * x, const std::string & name) {
        ggml_tensor * b = weight(name);
        return b ? ggml_add(ctx, x, b) : nullptr;
    }

    ggml_tensor * add_bias_spatial(ggml_tensor * x, const std::string & name) {
        ggml_tensor * b = weight(name);
        if (!b) return nullptr;
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        return ggml_add(ctx, x, b);
    }

    ggml_tensor * layer_norm(ggml_tensor * x, const std::string & prefix) {
        ggml_tensor * gamma = weight(prefix + "weight");
        ggml_tensor * beta = weight(prefix + "bias");
        if (!gamma || !beta) return nullptr;
        x = ggml_norm(ctx, x, 1e-5f);
        x = ggml_mul(ctx, x, gamma);
        return ggml_add(ctx, x, beta);
    }

    ggml_tensor * linear(ggml_tensor * x, const std::string & weight_name,
                         const std::string & bias_name = {}) {
        ggml_tensor * w = weight(weight_name);
        if (!w) return nullptr;
        ggml_tensor * out = ggml_mul_mat(ctx, w, x);
        ggml_mul_mat_set_prec(out, GGML_PREC_F32);
        return bias_name.empty() ? out : add_bias_tokens(out, bias_name);
    }

    ggml_tensor * conv2d(ggml_tensor * input, const std::string & prefix,
                         int stride, int pad) {
        ggml_tensor * w = weight(prefix + "weight");
        if (!w) return nullptr;
        const int64_t ow = (input->ne[0] + 2 * pad - w->ne[0]) / stride + 1;
        const int64_t oh = (input->ne[1] + 2 * pad - w->ne[1]) / stride + 1;
        ggml_tensor * args[] = {w, input};
        ggml_tensor * out = ggml_custom_4d(ctx, GGML_TYPE_F32, ow, oh, w->ne[3], input->ne[3],
            args, 2, nullptr, GGML_N_TASKS_MAX, nullptr);
        char custom_name[GGML_MAX_NAME];
        std::snprintf(custom_name, sizeof(custom_name), "rmbg_conv2d_s%d_p%d", stride, pad);
        ggml_set_name(out, custom_name);
        if (use_cuda_custom && ggml_backend_supports_op(backend, out)) {
            return weights.get_f32((prefix + "bias").c_str())
                ? add_bias_spatial(out, prefix + "bias") : out;
        }
        ggml_tensor * col = ggml_im2col(ctx, w, input, stride, stride, pad, pad,
                                        1, 1, true, GGML_TYPE_F32);
        out = ggml_mul_mat(
            ctx,
            ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2] * col->ne[3]),
            ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]));
        ggml_mul_mat_set_prec(out, GGML_PREC_F32);
        out = ggml_reshape_4d(ctx, out, col->ne[1], col->ne[2], col->ne[3], w->ne[3]);
        out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 1, 3, 2));
        return weights.get_f32((prefix + "bias").c_str())
            ? add_bias_spatial(out, prefix + "bias") : out;
    }

    ggml_tensor * batch_norm_impl(ggml_tensor * x, const std::string & prefix, bool relu) {
        const std::vector<float> * gamma = weights.get_f32((prefix + "weight").c_str());
        const std::vector<float> * beta = weights.get_f32((prefix + "bias").c_str());
        const std::vector<float> * mean = weights.get_f32((prefix + "running_mean").c_str());
        const std::vector<float> * var = weights.get_f32((prefix + "running_var").c_str());
        if (!gamma || !beta || !mean || !var || gamma->size() != beta->size() ||
            gamma->size() != mean->size() || gamma->size() != var->size()) return nullptr;
        std::vector<float> scale(gamma->size()), shift(gamma->size());
        for (size_t i = 0; i < gamma->size(); ++i) {
            scale[i] = (*gamma)[i] / std::sqrt((*var)[i] + 1e-5f);
            shift[i] = (*beta)[i] - (*mean)[i] * scale[i];
        }
        ggml_tensor * s = constant(GGML_TYPE_F32, {1, 1, (int64_t) scale.size(), 1},
                                   scale, "bn_scale");
        ggml_tensor * b = constant(GGML_TYPE_F32, {1, 1, (int64_t) shift.size(), 1},
                                   shift, "bn_shift");
        if (relu) {
            ggml_tensor * args[] = {x, s, b};
            ggml_tensor * fused = ggml_custom_4d(ctx, GGML_TYPE_F32,
                x->ne[0], x->ne[1], x->ne[2], x->ne[3], args, 3,
                nullptr, GGML_N_TASKS_MAX, nullptr);
            ggml_set_name(fused, "rmbg_affine_relu");
            if (use_backend_custom && ggml_backend_supports_op(backend, fused)) return fused;
        }
        x = ggml_add(ctx, ggml_mul(ctx, x, s), b);
        return relu ? ggml_relu(ctx, x) : x;
    }

    ggml_tensor * batch_norm(ggml_tensor * x, const std::string & prefix) {
        return batch_norm_impl(x, prefix, false);
    }

    ggml_tensor * batch_norm_relu(ggml_tensor * x, const std::string & prefix) {
        return batch_norm_impl(x, prefix, true);
    }

    ggml_tensor * resize(ggml_tensor * x, int H, int W) {
        return ggml_interpolate(ctx, x, W, H, x->ne[2], x->ne[3],
                                GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
    }

    ggml_tensor * add_scalar(ggml_tensor * x, float value) {
        return ggml_add(ctx, x, scalar(value));
    }

    ggml_tensor * valid_coordinate(ggml_tensor * x, int limit) {
        ggml_tensor * lo = ggml_step(ctx, add_scalar(x, 0.5f));
        ggml_tensor * hi = ggml_step(ctx, add_scalar(ggml_neg(ctx, x), (float) limit - 0.5f));
        return ggml_mul(ctx, lo, hi);
    }

    ggml_tensor * deform_sample_corner(ggml_tensor * input_tokens,
                                       ggml_tensor * yf, ggml_tensor * xf,
                                       ggml_tensor * weight_xy, int H, int W) {
        ggml_tensor * valid = ggml_mul(ctx, valid_coordinate(yf, H), valid_coordinate(xf, W));
        ggml_tensor * yc = ggml_clamp(ctx, ggml_dup(ctx, yf), 0.f, (float) H - 1.f);
        ggml_tensor * xc = ggml_clamp(ctx, ggml_dup(ctx, xf), 0.f, (float) W - 1.f);
        ggml_tensor * flat = ggml_add(ctx, xc, ggml_scale(ctx, yc, (float) W));
        ggml_tensor * index = ggml_cast(ctx, flat, GGML_TYPE_I32);
        ggml_tensor * values = ggml_get_rows(ctx, input_tokens, index);
        ggml_tensor * weight = ggml_reshape_2d(
            ctx, ggml_mul(ctx, weight_xy, valid), 1, ggml_nelements(weight_xy));
        return ggml_mul(ctx, values, weight);
    }

    static void deform_im2col_cpu(ggml_tensor * dst, int ith, int nth, void *) {
        const ggml_tensor * input = dst->src[0];
        const ggml_tensor * offset = dst->src[1];
        const ggml_tensor * modulator = dst->src[2];
        const int W = (int) input->ne[0];
        const int H = (int) input->ne[1];
        const int C = (int) input->ne[2];
        const int K = (int) modulator->ne[2];
        const int KW = (int) std::sqrt((float) K);
        const int KH = K / KW;
        const int64_t HW = (int64_t) H * W;
        const int64_t total = HW * C * K;
        const int64_t begin = total * ith / nth;
        const int64_t end = total * (ith + 1) / nth;
        const float * in = static_cast<const float *>(input->data);
        const float * off = static_cast<const float *>(offset->data);
        const float * mod = static_cast<const float *>(modulator->data);
        float * out = static_cast<float *>(dst->data);
        for (int64_t index = begin; index < end; ++index) {
            const int kc = (int) (index % (C * K));
            const int p = (int) (index / (C * K));
            const int k = kc % K;
            const int c = kc / K;
            const int yb = p / W;
            const int xb = p - yb * W;
            const float y = (float) (yb + k / KW - KH / 2) + off[p + HW * (2 * k)];
            const float x = (float) (xb + k % KW - KW / 2) + off[p + HW * (2 * k + 1)];
            const int y0 = (int) std::floor(y), x0 = (int) std::floor(x);
            const float dy = y - y0, dx = x - x0;
            float value = 0.f;
            auto add = [&](int sy, int sx, float scale) {
                if ((unsigned) sy < (unsigned) H && (unsigned) sx < (unsigned) W)
                    value += in[sx + (int64_t) W * sy + HW * c] * scale;
            };
            add(y0, x0, (1.f - dy) * (1.f - dx));
            add(y0, x0 + 1, (1.f - dy) * dx);
            add(y0 + 1, x0, dy * (1.f - dx));
            add(y0 + 1, x0 + 1, dy * dx);
            out[index] = value * (2.f / (1.f + std::exp(-mod[p + HW * k])));
        }
    }

    ggml_tensor * deform_im2col_legacy(ggml_tensor * input, ggml_tensor * offset,
                                       ggml_tensor * modulator, int H, int W,
                                       int C, int KH, int KW, int pad) {
        const int K = KH * KW;
        const int64_t HW = (int64_t) H * W;
        std::vector<float> base_y((size_t) HW), base_x((size_t) HW);
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                base_y[(size_t) h * W + w] = (float) h;
                base_x[(size_t) h * W + w] = (float) w;
            }
        ggml_tensor * by = constant(GGML_TYPE_F32, {HW}, base_y, "base_y");
        ggml_tensor * bx = constant(GGML_TYPE_F32, {HW}, base_x, "base_x");
        std::vector<float> kernel_y((size_t) K), kernel_x((size_t) K);
        for (int kh = 0; kh < KH; ++kh)
            for (int kw = 0; kw < KW; ++kw) {
                const int k = kh * KW + kw;
                kernel_y[(size_t) k] = (float) kh - pad;
                kernel_x[(size_t) k] = (float) kw - pad;
            }
        ggml_tensor * oy = ggml_view_2d(ctx, offset, HW, K,
                                        (size_t) 2 * HW * sizeof(float), 0);
        ggml_tensor * ox = ggml_view_2d(ctx, offset, HW, K,
                                        (size_t) 2 * HW * sizeof(float),
                                        (size_t) HW * sizeof(float));
        ggml_tensor * fy = ggml_add(ctx, ggml_add(ctx, oy, ggml_reshape_2d(ctx, by, HW, 1)),
                                    constant(GGML_TYPE_F32, {1, K}, kernel_y, "kernel_y"));
        ggml_tensor * fx = ggml_add(ctx, ggml_add(ctx, ox, ggml_reshape_2d(ctx, bx, HW, 1)),
                                    constant(GGML_TYPE_F32, {1, K}, kernel_x, "kernel_x"));
        ggml_tensor * y0 = ggml_floor(ctx, fy), * x0 = ggml_floor(ctx, fx);
        ggml_tensor * y1 = add_scalar(y0, 1.f), * x1 = add_scalar(x0, 1.f);
        ggml_tensor * dy = ggml_sub(ctx, fy, y0), * dx = ggml_sub(ctx, fx, x0);
        ggml_tensor * wy0 = add_scalar(ggml_neg(ctx, dy), 1.f);
        ggml_tensor * wx0 = add_scalar(ggml_neg(ctx, dx), 1.f);
        auto flat = [&](ggml_tensor * tensor) {
            return ggml_reshape_1d(ctx, tensor, HW * K);
        };
        ggml_tensor * tokens = spatial_to_tokens(input);
        ggml_tensor * s00 = deform_sample_corner(tokens, flat(y0), flat(x0),
            flat(ggml_mul(ctx, wy0, wx0)), H, W);
        ggml_tensor * s01 = deform_sample_corner(tokens, flat(y0), flat(x1),
            flat(ggml_mul(ctx, wy0, dx)), H, W);
        ggml_tensor * s10 = deform_sample_corner(tokens, flat(y1), flat(x0),
            flat(ggml_mul(ctx, dy, wx0)), H, W);
        ggml_tensor * s11 = deform_sample_corner(tokens, flat(y1), flat(x1),
            flat(ggml_mul(ctx, dy, dx)), H, W);
        ggml_tensor * samples = ggml_add(ctx, ggml_add(ctx, s00, s01),
                                         ggml_add(ctx, s10, s11));
        ggml_tensor * mask = ggml_reshape_2d(ctx, modulator, 1, HW * K);
        samples = ggml_mul(ctx, samples, ggml_scale(ctx, ggml_sigmoid(ctx, mask), 2.f));
        samples = ggml_reshape_3d(ctx, samples, C, HW, K);
        samples = ggml_cont(ctx, ggml_permute(ctx, samples, 1, 2, 0, 3));
        return ggml_reshape_2d(ctx, samples, (int64_t) C * K, HW);
    }

    ggml_tensor * deform_conv2d(ggml_tensor * input, const std::string & prefix) {
        ggml_tensor * regular = weight(prefix + "atrous_conv_regular_conv_weight");
        if (!regular || regular->ne[0] != regular->ne[1]) return nullptr;
        const int KW = (int) regular->ne[0];
        const int KH = (int) regular->ne[1];
        const int C = (int) input->ne[2];
        const int H = (int) input->ne[1];
        const int W = (int) input->ne[0];
        const int K = KH * KW;
        const int pad = KH > 1 ? KH / 2 : 0;
        const int64_t HW = (int64_t) H * W;
        ggml_tensor * offset = conv2d(input, prefix + "atrous_conv_offset_conv_", 1, pad);
        ggml_tensor * modulator = conv2d(input, prefix + "atrous_conv_modulator_conv_", 1, pad);
        if (!offset || !modulator) return nullptr;

        const std::string bias_name = prefix + "atrous_conv_regular_conv_bias";
        ggml_tensor * regular_bias = weight(bias_name);
        // RMBG's deformable regular projection has no checkpoint bias.  The
        // fused shader still accepts one so its writeback is uniform; a static
        // zero vector preserves the bias-free operator exactly.
        if (!regular_bias && regular) regular_bias = zeros(regular->ne[3]);
        const char * deform_project_env = std::getenv("RMBG_VK_DEFORM_PROJECT");
        const bool deform_project_enabled = deform_project_env &&
            std::strcmp(deform_project_env, "0") != 0;
        const bool deform_project_coop = deform_project_env &&
            std::strcmp(deform_project_env, "coop") == 0;
        // F16 exists only as a static CM multiplicand copy; F32 fallback
        // continues to use the source tensor without a conversion node.
        ggml_tensor * fused_regular = deform_project_enabled ? weight_f16(
            prefix + "atrous_conv_regular_conv_weight") : regular;
        ggml_tensor * fused_args[] = {input, offset, modulator, fused_regular, regular_bias};
        ggml_tensor * fused = fused_regular
            ? ggml_custom_4d(ctx, GGML_TYPE_F32, W, H, fused_regular->ne[3], 1,
                              fused_args, 5, nullptr, GGML_N_TASKS_MAX, nullptr)
            : nullptr;
        if (fused) {
            ggml_set_name(fused, deform_project_coop
                ? "rmbg_deform_project_cm1" : "rmbg_deform_project");
            const bool deform_project_supported = use_backend_custom &&
                ggml_backend_supports_op(backend, fused);
            if (deform_project_enabled && deform_project_supported) {
                return fused;
            }
        }

        ggml_tensor * args[] = {input, offset, modulator};
        ggml_tensor * stacked = ggml_custom_4d(ctx, GGML_TYPE_F32,
            (int64_t) C * K, HW, 1, 1, args, 3, deform_im2col_cpu,
            GGML_N_TASKS_MAX, nullptr);
        ggml_set_name(stacked, "rmbg_deform_im2col");
        if (!use_backend_custom || !ggml_backend_supports_op(backend, stacked)) {
            stacked = deform_im2col_legacy(input, offset, modulator, H, W, C, KH, KW, pad);
        }
        ggml_tensor * out = ggml_mul_mat(ctx,
            ggml_reshape_2d(ctx, regular, (int64_t) C * K, regular->ne[3]), stacked);
        ggml_mul_mat_set_prec(out, GGML_PREC_F32);
        out = tokens_to_spatial(out, H, W);
        return weights.get_f32(bias_name.c_str()) ? add_bias_spatial(out, bias_name) : out;
    }

    ggml_tensor * aspp_branch(ggml_tensor * input, const std::string & prefix) {
        ggml_tensor * x = deform_conv2d(input, prefix);
        return x ? batch_norm_relu(x, prefix + "bn_") : nullptr;
    }

    ggml_tensor * aspp(ggml_tensor * input, const std::string & prefix) {
        ggml_tensor * b1 = aspp_branch(input, prefix + "aspp1_");
        ggml_tensor * b2 = aspp_branch(input, prefix + "aspp_deforms_0_");
        ggml_tensor * b3 = aspp_branch(input, prefix + "aspp_deforms_1_");
        ggml_tensor * b4 = aspp_branch(input, prefix + "aspp_deforms_2_");
        if (!b1 || !b2 || !b3 || !b4) return nullptr;
        const int H = (int) input->ne[1], W = (int) input->ne[0];
        ggml_tensor * gap = ggml_pool_2d(ctx, input, GGML_OP_POOL_AVG,
                                        W, H, W, H, 0, 0);
        gap = conv2d(gap, prefix + "global_avg_pool_1_", 1, 0);
        gap = gap ? batch_norm_relu(gap, prefix + "global_avg_pool_2_") : nullptr;
        gap = gap ? resize(gap, H, W) : nullptr;
        if (!gap) return nullptr;
        ggml_tensor * cat = ggml_concat(ctx, b1, b2, 2);
        cat = ggml_concat(ctx, cat, b3, 2);
        cat = ggml_concat(ctx, cat, b4, 2);
        cat = ggml_concat(ctx, cat, gap, 2);
        ggml_tensor * out = conv2d(cat, prefix + "conv1_", 1, 0);
        return out ? batch_norm_relu(out, prefix + "bn1_") : nullptr;
    }

    ggml_tensor * basic_decoder_block(ggml_tensor * input, const std::string & prefix) {
        ggml_tensor * x = conv2d(input, prefix + "conv_in_", 1, 1);
        x = x ? batch_norm_relu(x, prefix + "bn_in_") : nullptr;
        x = x ? aspp(x, prefix + "dec_att_") : nullptr;
        x = x ? conv2d(x, prefix + "conv_out_", 1, 1) : nullptr;
        return x ? batch_norm(x, prefix + "bn_out_") : nullptr;
    }

    ggml_tensor * image_patches(ggml_tensor * image, int patch_h, int patch_w) {
        const int H = (int) image->ne[1], W = (int) image->ne[0];
        const int gh = H / patch_h, gw = W / patch_w;
        const int P = patch_h * patch_w;
        ggml_tensor * args[] = {image};
        ggml_tensor * gathered = ggml_custom_4d(ctx, GGML_TYPE_F32, 3, gh * gw * P, 1, 1,
            args, 1, nullptr, GGML_N_TASKS_MAX, nullptr);
        char fused_name[GGML_MAX_NAME];
        std::snprintf(fused_name, sizeof(fused_name),
                      "rmbg_image_patches_ph%d_pw%d", patch_h, patch_w);
        ggml_set_name(gathered, fused_name);
        if (!use_backend_custom || !ggml_backend_supports_op(backend, gathered)) {
            std::vector<int32_t> index((size_t) gh * gw * P);
            for (int hi = 0; hi < patch_h; ++hi)
                for (int wi = 0; wi < patch_w; ++wi) {
                    const int pos = hi * patch_w + wi;
                    for (int hg = 0; hg < gh; ++hg)
                        for (int wg = 0; wg < gw; ++wg) {
                            const int grid = hg * gw + wg;
                            index[(size_t) grid + (size_t) gh * gw * pos] =
                                (hg * patch_h + hi) * W + wg * patch_w + wi;
                        }
                }
            gathered = ggml_get_rows(ctx, spatial_to_tokens(image),
                constant(GGML_TYPE_I32, {(int64_t) index.size()}, index, "patch_idx"));
        }
        // get_rows produces [c, grid, pixel]. einops' "(c hg wg)" layout
        // requires grid to be the fastest varying component of the channel axis.
        gathered = ggml_reshape_3d(ctx, gathered, 3, gh * gw, P);
        gathered = ggml_cont(ctx, ggml_permute(ctx, gathered, 1, 0, 2, 3));
        gathered = ggml_reshape_2d(ctx, gathered, 3 * gh * gw, P);
        return tokens_to_spatial(gathered, patch_h, patch_w);
    }

    ggml_tensor * input_concat(ggml_tensor * image, ggml_tensor * feature,
                               const std::string & prefix) {
        ggml_tensor * patches = image_patches(image, (int) feature->ne[1], (int) feature->ne[0]);
        ggml_tensor * x = conv2d(patches, prefix + "conv1_", 1, 1);
        x = x ? conv2d(x, prefix + "conv_out_", 1, 1) : nullptr;
        return x ? ggml_concat(ctx, feature, x, 2) : nullptr;
    }

    ggml_tensor * gdt_attention(ggml_tensor * input, const std::string & gdt,
                                const std::string & attn) {
        ggml_tensor * gate = conv2d(input, gdt + "0_", 1, 1);
        gate = gate ? batch_norm_relu(gate, gdt + "1_") : nullptr;
        gate = gate ? conv2d(gate, attn + "0_", 1, 0) : nullptr;
        gate = gate ? ggml_sigmoid(ctx, gate) : nullptr;
        return gate ? ggml_mul(ctx, input, gate) : nullptr;
    }

    EncoderTaps encoder_4scale(ggml_tensor * input, int H, int W, SwinTaps & full_debug) {
        EncoderTaps out;
        full_debug = backbone(input, H, W);
        ggml_tensor * half_input = resize(input, H / 2, W / 2);
        SwinTaps half = backbone(half_input, H / 2, W / 2);
        if (!full_debug.x1 || !half.x1) return out;
        out.x1 = ggml_concat(ctx, full_debug.x1, resize(half.x1, H / 4, W / 4), 2);
        out.x2 = ggml_concat(ctx, full_debug.x2, resize(half.x2, H / 8, W / 8), 2);
        out.x3 = ggml_concat(ctx, full_debug.x3, resize(half.x3, H / 16, W / 16), 2);
        out.x4 = ggml_concat(ctx, full_debug.x4, resize(half.x4, H / 32, W / 32), 2);
        ggml_tensor * ctx4 = resize(out.x1, H / 32, W / 32);
        ctx4 = ggml_concat(ctx, ctx4, resize(out.x2, H / 32, W / 32), 2);
        ctx4 = ggml_concat(ctx, ctx4, resize(out.x3, H / 32, W / 32), 2);
        out.x4 = ggml_concat(ctx, ctx4, out.x4, 2);
        return out;
    }

    ggml_tensor * decoder(ggml_tensor * image, const EncoderTaps & enc, int H, int W) {
        ggml_tensor * x4 = basic_decoder_block(enc.x4, "sq0_");
        if (!x4) return nullptr;
        ggml_tensor * p4 = input_concat(image, x4, "ipt5_");
        p4 = p4 ? basic_decoder_block(p4, "db4_") : nullptr;
        p4 = p4 ? gdt_attention(p4, "gdt4_", "gdta4_") : nullptr;
        if (!p4) return nullptr;
        ggml_tensor * p3 = ggml_add(ctx, resize(p4, H / 16, W / 16),
                                    conv2d(enc.x3, "lat4_conv_", 1, 0));
        p3 = input_concat(image, p3, "ipt4_");
        p3 = p3 ? basic_decoder_block(p3, "db3_") : nullptr;
        p3 = p3 ? gdt_attention(p3, "gdt3_", "gdta3_") : nullptr;
        if (!p3) return nullptr;
        ggml_tensor * p2 = ggml_add(ctx, resize(p3, H / 8, W / 8),
                                    conv2d(enc.x2, "lat3_conv_", 1, 0));
        p2 = input_concat(image, p2, "ipt3_");
        p2 = p2 ? basic_decoder_block(p2, "db2_") : nullptr;
        p2 = p2 ? gdt_attention(p2, "gdt2_", "gdta2_") : nullptr;
        if (!p2) return nullptr;
        ggml_tensor * p1 = ggml_add(ctx, resize(p2, H / 4, W / 4),
                                    conv2d(enc.x1, "lat2_conv_", 1, 0));
        p1 = input_concat(image, p1, "ipt2_");
        p1 = p1 ? basic_decoder_block(p1, "db1_") : nullptr;
        p1 = p1 ? resize(p1, H, W) : nullptr;
        p1 = p1 ? input_concat(image, p1, "ipt1_") : nullptr;
        ggml_tensor * logits = p1 ? conv2d(p1, "out1_", 1, 0) : nullptr;
        return logits ? ggml_sigmoid(ctx, logits) : nullptr;
    }

    ggml_tensor * spatial_to_tokens(ggml_tensor * x) {
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
        return ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2] * x->ne[3]);
    }

    ggml_tensor * tokens_to_spatial(ggml_tensor * x, int H, int W) {
        x = ggml_reshape_4d(ctx, x, x->ne[0], W, H, 1);
        return ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
    }

    static void build_shift_mask(int H, int W, int ws, int shift,
                                 std::vector<float> & out) {
        const int Hp = H + (ws - H % ws) % ws;
        const int Wp = W + (ws - W % ws) % ws;
        const int nW = (Hp / ws) * (Wp / ws);
        const int N = ws * ws;
        std::vector<int> regions((size_t) Hp * Wp, 0);
        const int hb[4] = {0, Hp - ws, Hp - shift, Hp};
        const int wb[4] = {0, Wp - ws, Wp - shift, Wp};
        int region = 0;
        for (int hi = 0; hi < 3; ++hi) {
            for (int wi = 0; wi < 3; ++wi) {
                for (int h = hb[hi]; h < hb[hi + 1]; ++h)
                    for (int w = wb[wi]; w < wb[wi + 1]; ++w)
                        regions[(size_t) h * Wp + w] = region;
                ++region;
            }
        }
        out.assign((size_t) N * N * nW, 0.f);
        const int windows_w = Wp / ws;
        for (int win = 0; win < nW; ++win) {
            const int wh = (win / windows_w) * ws;
            const int ww = (win % windows_w) * ws;
            for (int q = 0; q < N; ++q) {
                const int qr = regions[(size_t) (wh + q / ws) * Wp + ww + q % ws];
                for (int k = 0; k < N; ++k) {
                    const int kr = regions[(size_t) (wh + k / ws) * Wp + ww + k % ws];
                    out[(size_t) k + (size_t) N * (q + (size_t) N * win)] =
                        qr == kr ? 0.f : -100.f;
                }
            }
        }
    }

    ggml_tensor * swin_block(ggml_tensor * tokens, int H, int W, int stage, int block,
                             SwinTaps * taps = nullptr) {
        const std::string p = "bb_layers_" + std::to_string(stage) + "_blocks_" +
                              std::to_string(block) + "_";
        ggml_tensor * shortcut = tokens;
        tokens = layer_norm(tokens, p + "norm1_");
        if (!tokens) return nullptr;
        if (taps && stage == 0 && block == 0) taps->block0_norm1 = tokens;
        const int C = (int) tokens->ne[0];
        constexpr int ws = 12;
        const int shift = block % 2 ? ws / 2 : 0;
        const int Hp = H + (ws - H % ws) % ws;
        const int Wp = W + (ws - W % ws) % ws;
        const int nW = (Hp / ws) * (Wp / ws);
        const int N = ws * ws;
        const int L = H * W;

        const std::vector<float> * table =
            weights.get_f32((p + "attn_relative_position_bias_table").c_str());
        const std::vector<float> * index =
            weights.get_f32((p + "attn_relative_position_index").c_str());
        if (!table || !index || index->size() != (size_t) N * N || table->size() % 529 != 0) {
            return nullptr;
        }
        const int heads = (int) (table->size() / 529);
        const int hd = C / heads;
        if (hd != 32) return nullptr;
        std::vector<float> rpb((size_t) N * N * heads);
        for (int h = 0; h < heads; ++h)
            for (int qi = 0; qi < N; ++qi)
                for (int ki = 0; ki < N; ++ki)
                    rpb[(size_t) ki + (size_t) N * (qi + (size_t) N * h)] =
                        (*table)[(size_t) (int) (*index)[(size_t) qi * N + ki] * heads + h];
        // CUDA's existing fused node consumes this F32 RPB tensor. Vulkan
        // folds the same values into its Flash Attention mask below.
        ggml_tensor * rpb_tensor =
            constant(GGML_TYPE_F32, {N, N, heads, 1}, rpb, "rpb");
        ggml_tensor * qkv_weight = weight(p + "attn_qkv_weight");
        ggml_tensor * qkv_bias = weight(p + "attn_qkv_bias");
        ggml_tensor * proj_weight = weight(p + "attn_proj_weight");
        ggml_tensor * proj_bias = weight(p + "attn_proj_bias");
        if (!qkv_weight || !qkv_bias || !proj_weight || !proj_bias) return nullptr;

        ggml_tensor * attention_input = tokens;
        ggml_tensor * args[] = {
            attention_input, qkv_weight, qkv_bias, proj_weight, proj_bias, rpb_tensor,
        };
        ggml_tensor * fused = ggml_custom_4d(ctx, GGML_TYPE_F32, C, L, 1, 1,
            args, 6, nullptr, GGML_N_TASKS_MAX, nullptr);
        char fused_name[GGML_MAX_NAME];
        std::snprintf(fused_name, sizeof(fused_name),
                      "rmbg_swin_attn_h%d_w%d_sh%d", H, W, shift);
        ggml_set_name(fused, fused_name);
        const bool can_fuse = use_cuda_custom && ggml_backend_supports_op(backend, fused);
        const bool need_debug_path = taps && stage == 0 && block == 0;

        if (can_fuse) {
            tokens = fused;
        }
        if (!can_fuse || need_debug_path) {
            ggml_tensor * pack_args[] = {attention_input};
            ggml_tensor * windows = ggml_custom_4d(ctx, GGML_TYPE_F32, C, N, nW, 1,
                pack_args, 1, nullptr, GGML_N_TASKS_MAX, nullptr);
            char pack_name[GGML_MAX_NAME];
            std::snprintf(pack_name, sizeof(pack_name), "rmbg_swin_pack_h%d_w%d_sh%d", H, W, shift);
            ggml_set_name(windows, pack_name);
            if (!use_backend_custom || !ggml_backend_supports_op(backend, windows)) {
                std::vector<int32_t> partition((size_t) N * nW);
                const int windows_w = Wp / ws;
                for (int win = 0; win < nW; ++win) {
                    const int wh = (win / windows_w) * ws;
                    const int ww = (win % windows_w) * ws;
                    for (int t = 0; t < N; ++t) {
                        const int rh = wh + t / ws;
                        const int rw = ww + t % ws;
                        const int sh = (rh + shift) % Hp;
                        const int sw = (rw + shift) % Wp;
                        partition[(size_t) t + (size_t) N * win] =
                            (sh < H && sw < W) ? sh * W + sw : L;
                    }
                }
                ggml_tensor * partition_idx =
                    constant(GGML_TYPE_I32, {N * nW}, partition, "win_idx");
                ggml_tensor * padded_tokens = ggml_concat(ctx, attention_input, zeros(C), 1);
                windows = ggml_get_rows(ctx, padded_tokens, partition_idx);
                windows = ggml_reshape_3d(ctx, windows, C, N, nW);
            }
            if (taps && stage == 0 && block == 0) taps->block0_windows = windows;

            ggml_tensor * qkv = ggml_mul_mat(ctx, qkv_weight, windows);
            ggml_mul_mat_set_prec(qkv, GGML_PREC_F32);

            // Vulkan consumes the QKV projection directly in the layout used by
            // attention. This removes the broadcast add and six materialized
            // view/permute copies without changing the F32 attention contract.
            ggml_tensor * qkv_args[] = {qkv, qkv_bias};
            ggml_tensor * q = ggml_custom_4d(ctx, GGML_TYPE_F32, hd, N, heads, nW,
                qkv_args, 2, nullptr, GGML_N_TASKS_MAX, nullptr);
            ggml_tensor * k = ggml_custom_4d(ctx, GGML_TYPE_F32, hd, N, heads, nW,
                qkv_args, 2, nullptr, GGML_N_TASKS_MAX, nullptr);
            ggml_tensor * v = ggml_custom_4d(ctx, GGML_TYPE_F32, hd, N, heads, nW,
                qkv_args, 2, nullptr, GGML_N_TASKS_MAX, nullptr);
            ggml_set_name(q, "rmbg_swin_qkv_layout_k0");
            ggml_set_name(k, "rmbg_swin_qkv_layout_k1");
            ggml_set_name(v, "rmbg_swin_qkv_layout_k2");
            const char * qkv_layout_env = std::getenv("RMBG_VK_QKV_LAYOUT");
            const bool qkv_layout_enabled = !qkv_layout_env || std::strcmp(qkv_layout_env, "0") != 0;
            const bool can_fuse_qkv_layout = qkv_layout_enabled && use_backend_custom &&
                ggml_backend_supports_op(backend, q) &&
                ggml_backend_supports_op(backend, k) &&
                ggml_backend_supports_op(backend, v);
            if (!can_fuse_qkv_layout) {
                qkv = ggml_add(ctx, qkv, qkv_bias);
                q = ggml_view_3d(ctx, qkv, C, N, nW, qkv->nb[1], qkv->nb[2], 0);
                k = ggml_view_3d(ctx, qkv, C, N, nW, qkv->nb[1], qkv->nb[2],
                                  (size_t) C * sizeof(float));
                v = ggml_view_3d(ctx, qkv, C, N, nW, qkv->nb[1], qkv->nb[2],
                                  (size_t) 2 * C * sizeof(float));
                q = ggml_cont(ctx, q);
                k = ggml_cont(ctx, k);
                v = ggml_cont(ctx, v);
                q = ggml_reshape_4d(ctx, q, hd, heads, N, nW);
                k = ggml_reshape_4d(ctx, k, hd, heads, N, nW);
                v = ggml_reshape_4d(ctx, v, hd, heads, N, nW);
                q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
                k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
                v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
            }

            // Flash Attention consumes the complete shifted-window mask.  The
            // mask stores relative-position bias and region blocking together,
            // so QK, softmax, and AV never materialize an N*N score tensor.
            std::vector<ggml_fp16_t> mask_f16((size_t) N * N * heads * nW);
            const int windows_w = Wp / ws;
            for (int win = 0; win < nW; ++win) {
                const int wh = (win / windows_w) * ws;
                const int ww = (win % windows_w) * ws;
                for (int head = 0; head < heads; ++head) {
                    for (int qi = 0; qi < N; ++qi) {
                        const int qr = (wh + qi / ws);
                        const int qc = (ww + qi % ws);
                        const int qregion = shift == 0 ? 0 :
                            ((qr < Hp - ws ? 0 : (qr < Hp - shift ? 1 : 2)) * 3 +
                             (qc < Wp - ws ? 0 : (qc < Wp - shift ? 1 : 2)));
                        for (int ki = 0; ki < N; ++ki) {
                            const int kr = wh + ki / ws;
                            const int kc = ww + ki % ws;
                            const int kregion = shift == 0 ? 0 :
                                ((kr < Hp - ws ? 0 : (kr < Hp - shift ? 1 : 2)) * 3 +
                                 (kc < Wp - ws ? 0 : (kc < Wp - shift ? 1 : 2)));
                            float value = rpb[(size_t) ki + (size_t) N *
                                               (qi + (size_t) N * head)];
                            if (shift && qregion != kregion) value = -100.f;
                            mask_f16[(size_t) ki + (size_t) N *
                                     (qi + (size_t) N *
                                      (head + (size_t) heads * win))] =
                                ggml_fp32_to_fp16(value);
                        }
                    }
                }
            }
            ggml_tensor * swin_mask = constant(GGML_TYPE_F16,
                {N, N, heads, nW}, mask_f16, "swin_mask");
            // The scalar Flash path accepts F32 Q/K/V and accumulates in F32.
            // Cooperative matrices need F16 K/V on current Vulkan devices, so
            // they are an explicit opt-in until the endpoint parity gate says
            // their input rounding is acceptable for a given model.
            const bool disable_vk_f16 = std::getenv("GGML_VK_DISABLE_F16") != nullptr;
            const char * flash_env = std::getenv("RMBG_VK_FLASH_ATTN");
            const bool flash_disabled = flash_env && std::strcmp(flash_env, "0") == 0;
            // `coop`, or `coop0` ... `coop3`, enables the F16 cooperative
            // kernel for all or one Swin stage.  The staged spelling makes it
            // possible to allocate a model's endpoint-error budget from data,
            // rather than assuming every attention block is equally tolerant.
            const bool coop_requested = flash_env &&
                std::strncmp(flash_env, "coop", 4) == 0;
            const int coop_stage = coop_requested && flash_env[4] != '\0'
                ? std::atoi(flash_env + 4) : -1;
            const bool flash_coop = coop_requested &&
                (coop_stage < 0 || coop_stage == stage);
            // F32 scalar Flash remains valid in strict mode.  Only the
            // cooperative variant requires the device F16 path.
            const bool use_flash = use_vulkan_custom && flash_env && !flash_disabled &&
                (!disable_vk_f16 || !flash_coop);
            ggml_tensor * k_attn = use_flash && flash_coop ? ggml_cast(ctx, k, GGML_TYPE_F16) : k;
            ggml_tensor * v_attn = use_flash && flash_coop ? ggml_cast(ctx, v, GGML_TYPE_F16) : v;
            ggml_tensor * attended = nullptr;
            if (use_flash) {
                ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k_attn, v_attn, swin_mask,
                    1.f / std::sqrt((float) hd), 0.f, 0.f);
                ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
                // Flash output is contiguous [head_dim, head, token, window].
                // Reshaping merges lane and head directly into the projection's
                // channel axis; permuting here would scramble head ownership.
                attended = ggml_reshape_3d(ctx, attn, C, N, nW);
            } else {
                ggml_tensor * attn_scores = ggml_mul_mat(
                    ctx, k, ggml_scale(ctx, q, 1.f / std::sqrt((float) hd)));
                ggml_mul_mat_set_prec(attn_scores, GGML_PREC_F32);
                attn_scores = ggml_add(ctx, attn_scores,
                    constant(GGML_TYPE_F32, {N, N, heads, 1}, rpb, "rpb_strict"));
                if (shift) {
                    std::vector<float> strict_mask;
                    build_shift_mask(H, W, ws, shift, strict_mask);
                    attn_scores = ggml_add(ctx, attn_scores,
                        constant(GGML_TYPE_F32, {N, N, 1, nW}, strict_mask, "mask_strict"));
                }
                attn_scores = ggml_soft_max(ctx, attn_scores);
                ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, v));
                attended = ggml_mul_mat(ctx, vt, attn_scores);
                ggml_mul_mat_set_prec(attended, GGML_PREC_F32);
                attended = ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3));
                attended = ggml_reshape_3d(ctx, attended, C, N, nW);
            }
            windows = ggml_mul_mat(ctx, proj_weight, attended);
            ggml_mul_mat_set_prec(windows, GGML_PREC_F32);
            windows = ggml_add(ctx, windows, proj_bias);
            if (taps && stage == 0 && block == 0) taps->block0_attended_windows = windows;

            ggml_tensor * unpack_args[] = {windows};
            ggml_tensor * primitive_tokens = ggml_custom_4d(ctx, GGML_TYPE_F32, C, L, 1, 1,
                unpack_args, 1, nullptr, GGML_N_TASKS_MAX, nullptr);
            char unpack_name[GGML_MAX_NAME];
            std::snprintf(unpack_name, sizeof(unpack_name), "rmbg_swin_unpack_h%d_w%d_sh%d", H, W, shift);
            ggml_set_name(primitive_tokens, unpack_name);
            if (!use_backend_custom || !ggml_backend_supports_op(backend, primitive_tokens)) {
                std::vector<int32_t> inverse(L);
                const int windows_w = Wp / ws;
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        const int rh = (h - shift + Hp) % Hp;
                        const int rw = (w - shift + Wp) % Wp;
                        const int win = (rh / ws) * windows_w + rw / ws;
                        const int t = (rh % ws) * ws + rw % ws;
                        inverse[(size_t) h * W + w] = win * N + t;
                    }
                }
                ggml_tensor * flat_windows = ggml_reshape_2d(ctx, windows, C, N * nW);
                primitive_tokens = ggml_get_rows(ctx, flat_windows,
                    constant(GGML_TYPE_I32, {L}, inverse, "win_inv"));
            }
            if (!can_fuse) tokens = primitive_tokens;
        }
        tokens = ggml_add(ctx, tokens, shortcut);
        if (taps && stage == 0 && block == 0) taps->block0_after_attention = tokens;

        shortcut = tokens;
        tokens = layer_norm(tokens, p + "norm2_");
        tokens = linear(tokens, p + "mlp_fc1_weight", p + "mlp_fc1_bias");
        if (!tokens) return nullptr;
        tokens = ggml_gelu_erf(ctx, tokens);
        tokens = linear(tokens, p + "mlp_fc2_weight", p + "mlp_fc2_bias");
        return tokens ? ggml_add(ctx, tokens, shortcut) : nullptr;
    }

    ggml_tensor * patch_merge(ggml_tensor * tokens, int H, int W, int stage) {
        const int rows = (H / 2) * (W / 2);
        const int C = (int) tokens->ne[0];
        const std::string p = "bb_layers_" + std::to_string(stage) + "_downsample_";
        ggml_tensor * args[] = {tokens};
        ggml_tensor * fused = ggml_custom_4d(ctx, GGML_TYPE_F32, C * 4, rows, 1, 1,
            args, 1, nullptr, GGML_N_TASKS_MAX, nullptr);
        char fused_name[GGML_MAX_NAME];
        std::snprintf(fused_name, sizeof(fused_name), "rmbg_patch_merge_h%d_w%d", H, W);
        ggml_set_name(fused, fused_name);
        ggml_tensor * merged = nullptr;
        if (use_backend_custom && ggml_backend_supports_op(backend, fused)) {
            merged = fused;
        } else {
            std::vector<int32_t> idx((size_t) 4 * rows);
            for (int oh = 0; oh < H / 2; ++oh) {
                for (int ow = 0; ow < W / 2; ++ow) {
                    const int row = oh * (W / 2) + ow;
                    idx[0 + (size_t) 4 * row] = (oh * 2) * W + ow * 2;
                    idx[1 + (size_t) 4 * row] = (oh * 2 + 1) * W + ow * 2;
                    idx[2 + (size_t) 4 * row] = (oh * 2) * W + ow * 2 + 1;
                    idx[3 + (size_t) 4 * row] = (oh * 2 + 1) * W + ow * 2 + 1;
                }
            }
            merged = ggml_get_rows(ctx, tokens,
                constant(GGML_TYPE_I32, {4 * rows}, idx, "merge_idx"));
            merged = ggml_reshape_2d(ctx, merged, tokens->ne[0] * 4, rows);
        }
        merged = layer_norm(merged, p + "norm_");
        return merged ? linear(merged, p + "reduction_weight") : nullptr;
    }

    SwinTaps backbone(ggml_tensor * input, int H, int W) {
        SwinTaps taps;
        ggml_tensor * x = conv2d(input, "bb_patch_embed_proj_", 4, 0);
        if (!x) return taps;
        x = spatial_to_tokens(x);
        taps.patch_pre_norm = x;
        x = layer_norm(x, "bb_patch_embed_norm_");
        taps.patch = x;
        static const int depths[4] = {2, 2, 18, 2};
        int stage_h = H / 4;
        int stage_w = W / 4;
        for (int stage = 0; stage < 4; ++stage) {
            for (int block = 0; block < depths[stage]; ++block) {
                x = swin_block(x, stage_h, stage_w, stage, block, &taps);
                if (!x) return {};
                if (stage == 0 && block == 0) taps.block0 = x;
            }
            taps.raw[stage] = x;
            ggml_tensor * normed = layer_norm(x, "bb_norm" + std::to_string(stage) + "_");
            ggml_tensor * spatial = normed ? tokens_to_spatial(normed, stage_h, stage_w) : nullptr;
            if (stage == 0) taps.x1 = spatial;
            if (stage == 1) taps.x2 = spatial;
            if (stage == 2) taps.x3 = spatial;
            if (stage == 3) taps.x4 = spatial;
            if (stage < 3) {
                x = patch_merge(x, stage_h, stage_w, stage);
                stage_h /= 2;
                stage_w /= 2;
            }
        }
        return taps;
    }
};

} // namespace

struct RmbgDeviceGraph::Impl {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx_static = nullptr;
    ggml_context * ctx_compute = nullptr;
    ggml_backend_buffer_t static_buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * alpha = nullptr;
    SwinTaps encoder;
    EncoderTaps encoder4;
    int input_size = 0;
    size_t compute_size = 0;

    ~Impl() {
        if (allocator) ggml_gallocr_free(allocator);
        if (static_buffer) ggml_backend_buffer_free(static_buffer);
        if (ctx_compute) ggml_free(ctx_compute);
        if (ctx_static) ggml_free(ctx_static);
    }
};

RmbgDeviceGraph::RmbgDeviceGraph() : impl_(new Impl) {}
RmbgDeviceGraph::~RmbgDeviceGraph() = default;

bool RmbgDeviceGraph::init(ggml_backend_t backend, const WeightMap & weights, int input_size,
                           std::string & err) {
    impl_.reset(new Impl);
    if (!backend) { err = "null backend"; return false; }
    if (input_size <= 0 || input_size % 32 != 0) {
        err = "input size must be positive and divisible by 32";
        return false;
    }
    impl_->backend = backend;
    impl_->input_size = input_size;

    ggml_init_params ps{ggml_tensor_overhead() * 8192, nullptr, true};
    ggml_init_params pc{
        ggml_tensor_overhead() * kMaxGraphNodes + ggml_graph_overhead_custom(kMaxGraphNodes, false),
        nullptr, true};
    impl_->ctx_static = ggml_init(ps);
    impl_->ctx_compute = ggml_init(pc);
    if (!impl_->ctx_static || !impl_->ctx_compute) {
        err = "ggml context allocation failed";
        return false;
    }

    GraphBuilder b(impl_->ctx_static, impl_->ctx_compute, backend, weights);
    impl_->input = ggml_new_tensor_4d(impl_->ctx_compute, GGML_TYPE_F32,
                                      input_size, input_size, 3, 1);
    ggml_set_name(impl_->input, "rmbg_input");
    ggml_set_input(impl_->input);
    impl_->encoder4 = b.encoder_4scale(impl_->input, input_size, input_size, impl_->encoder);
    if (!impl_->encoder.x1 || !impl_->encoder.x2 || !impl_->encoder.x3 || !impl_->encoder.x4) {
        err = "failed to build Swin graph: missing weights or invalid tensor shape";
        return false;
    }

    if (weights.get_f32("sq0_conv_in_weight")) {
        impl_->alpha = b.decoder(impl_->input, impl_->encoder4, input_size, input_size);
        if (!impl_->alpha) {
            err = "failed to build decoder graph: missing weights or invalid tensor shape";
            return false;
        }
        ggml_set_name(impl_->alpha, "rmbg_alpha");
        ggml_set_output(impl_->alpha);
    }
    impl_->graph = ggml_new_graph_custom(impl_->ctx_compute, kMaxGraphNodes, false);
    if (impl_->alpha) {
        // Production graph: alpha is the only retained output, allowing gallocr
        // to reuse all encoder and decoder intermediates.
        ggml_build_forward_expand(impl_->graph, impl_->alpha);
    } else {
        // Encoder-only parity graph: retain validation taps as graph outputs.
        ggml_tensor * debug_outputs[] = {
            impl_->encoder.x1, impl_->encoder.x2, impl_->encoder.x3, impl_->encoder.x4,
            impl_->encoder.patch, impl_->encoder.patch_pre_norm, impl_->encoder.block0,
            impl_->encoder.block0_norm1, impl_->encoder.block0_windows,
            impl_->encoder.block0_attended_windows, impl_->encoder.block0_after_attention,
        };
        for (ggml_tensor * tensor : debug_outputs) {
            ggml_set_output(tensor);
            ggml_build_forward_expand(impl_->graph, tensor);
        }
        for (ggml_tensor * raw : impl_->encoder.raw) {
            ggml_set_output(raw);
            ggml_build_forward_expand(impl_->graph, raw);
        }
    }

    impl_->static_buffer = ggml_backend_alloc_ctx_tensors(impl_->ctx_static, backend);
    if (!impl_->static_buffer) { err = "static backend buffer allocation failed"; return false; }
    for (const StaticBlob & blob : b.blobs) {
        if (blob.bytes.size() != ggml_nbytes(blob.tensor)) {
            err = std::string("static tensor byte mismatch: ") + ggml_get_name(blob.tensor);
            return false;
        }
        ggml_backend_tensor_set(blob.tensor, blob.bytes.data(), 0, blob.bytes.size());
    }

    impl_->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!impl_->allocator || !ggml_gallocr_reserve(impl_->allocator, impl_->graph) ||
        !ggml_gallocr_alloc_graph(impl_->allocator, impl_->graph)) {
        err = "compute graph allocation failed";
        return false;
    }
    impl_->compute_size = ggml_gallocr_get_buffer_size(impl_->allocator, 0);
    return true;
}

bool RmbgDeviceGraph::forward_encoder(const std::vector<float> & input_nchw,
                                      std::vector<float> & x1, std::vector<float> & x2,
                                      std::vector<float> & x3, std::vector<float> & x4,
                                      std::string & err) {
    if (!impl_->graph || !impl_->input) { err = "graph not initialized"; return false; }
    const size_t expected = (size_t) 3 * impl_->input_size * impl_->input_size;
    if (input_nchw.size() != expected) { err = "input tensor size mismatch"; return false; }
    ggml_backend_tensor_set(impl_->input, input_nchw.data(), 0, expected * sizeof(float));
    if (ggml_backend_graph_compute(impl_->backend, impl_->graph) != GGML_STATUS_SUCCESS) {
        err = "backend graph compute failed";
        return false;
    }
    auto read = [](ggml_tensor * tensor, std::vector<float> & out) {
        out.resize((size_t) ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(float));
    };
    read(impl_->encoder.x1, x1);
    read(impl_->encoder.x2, x2);
    read(impl_->encoder.x3, x3);
    read(impl_->encoder.x4, x4);
    return true;
}

bool RmbgDeviceGraph::forward(const std::vector<float> & input_nchw,
                              std::vector<float> & alpha, std::string & err) {
    if (!impl_->alpha) { err = "decoder graph not built"; return false; }
    const size_t expected = (size_t) 3 * impl_->input_size * impl_->input_size;
    if (input_nchw.size() != expected) { err = "input tensor size mismatch"; return false; }
    ggml_backend_tensor_set(impl_->input, input_nchw.data(), 0, expected * sizeof(float));
    if (ggml_backend_graph_compute(impl_->backend, impl_->graph) != GGML_STATUS_SUCCESS) {
        err = "backend graph compute failed";
        return false;
    }
    alpha.resize((size_t) ggml_nelements(impl_->alpha));
    ggml_backend_tensor_get(impl_->alpha, alpha.data(), 0, alpha.size() * sizeof(float));
    return true;
}

bool RmbgDeviceGraph::forward_swin_debug(const std::vector<float> & input_nchw,
                                         std::vector<float> & patch_tokens,
                                         std::vector<float> & block0_tokens,
                                         std::vector<float> & stage0_tokens,
                                         std::vector<float> & stage1_tokens,
                                         std::vector<float> & stage2_tokens,
                                         std::vector<float> & stage3_tokens,
                                         std::string & err) {
    std::vector<float> unused1, unused2, unused3, unused4;
    if (!forward_encoder(input_nchw, unused1, unused2, unused3, unused4, err)) return false;
    auto read = [](ggml_tensor * tensor, std::vector<float> & out) {
        out.resize((size_t) ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(float));
    };
    read(impl_->encoder.patch, patch_tokens);
    read(impl_->encoder.block0, block0_tokens);
    read(impl_->encoder.raw[0], stage0_tokens);
    read(impl_->encoder.raw[1], stage1_tokens);
    read(impl_->encoder.raw[2], stage2_tokens);
    read(impl_->encoder.raw[3], stage3_tokens);
    return true;
}

bool RmbgDeviceGraph::forward_block0_debug(const std::vector<float> & input_nchw,
                                           std::vector<float> & patch_pre_norm,
                                           std::vector<float> & norm1,
                                           std::vector<float> & window0,
                                           std::vector<float> & attended_window0,
                                           std::vector<float> & after_attention,
                                           std::string & err) {
    std::vector<float> x1, x2, x3, x4;
    if (!forward_encoder(input_nchw, x1, x2, x3, x4, err)) return false;
    auto read = [](ggml_tensor * tensor, std::vector<float> & out, size_t count = 0) {
        if (!count) count = (size_t) ggml_nelements(tensor);
        out.resize(count);
        ggml_backend_tensor_get(tensor, out.data(), 0, count * sizeof(float));
    };
    read(impl_->encoder.patch_pre_norm, patch_pre_norm);
    read(impl_->encoder.block0_norm1, norm1);
    const size_t window_elements = 192 * 12 * 12;
    read(impl_->encoder.block0_windows, window0, window_elements);
    read(impl_->encoder.block0_attended_windows, attended_window0, window_elements);
    read(impl_->encoder.block0_after_attention, after_attention);
    return true;
}

size_t RmbgDeviceGraph::compute_bytes() const { return impl_->compute_size; }

} // namespace rmbg
