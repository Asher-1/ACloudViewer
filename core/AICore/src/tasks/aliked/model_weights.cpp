// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "model_weights.hpp"

#include <ggml.h>
#include <gguf.h>

#include <cstring>

namespace lightglue::aliked_internal {
namespace {

constexpr const char *kArchitecture = "aliked";

std::vector<float> CopyTensor(const ggml_tensor *tensor, std::string *error) {
    if (tensor == nullptr) {
        *error = "null tensor";
        return {};
    }
    const size_t count = ggml_nelements(tensor);
    std::vector<float> data(count);
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(data.data(), tensor->data, count * sizeof(float));
        return data;
    }
    if (tensor->type == GGML_TYPE_F16) {
        const ggml_fp16_t *src = static_cast<const ggml_fp16_t *>(tensor->data);
        for (size_t i = 0; i < count; ++i) {
            data[i] = ggml_fp16_to_fp32(src[i]);
        }
        return data;
    }
    if (tensor->type == GGML_TYPE_Q8_0) {
        // Q8_0 block: { ggml_fp16_t d (scale), int8_t qs[32] }
        const size_t block_bytes = ggml_type_size(GGML_TYPE_Q8_0);
        const int64_t block_elems = ggml_blck_size(GGML_TYPE_Q8_0);
        const size_t num_blocks = count / block_elems;
        const uint8_t *raw = static_cast<const uint8_t *>(tensor->data);
        for (size_t b = 0; b < num_blocks; ++b) {
            const uint8_t *bp = raw + b * block_bytes;
            ggml_fp16_t d_half;
            std::memcpy(&d_half, bp, sizeof(ggml_fp16_t));
            const float scale = ggml_fp16_to_fp32(d_half);
            const int8_t *qs =
                    reinterpret_cast<const int8_t *>(bp + sizeof(ggml_fp16_t));
            const size_t base = b * block_elems;
            for (int64_t j = 0; j < block_elems; ++j) {
                data[base + j] = scale * qs[j];
            }
        }
        return data;
    }
    *error = "unsupported tensor element type";
    return {};
}

}  // namespace

bool LoadAlikedTensors(const std::string &path,
                       TensorMap *tensors,
                       int32_t *descriptor_dim,
                       std::string *error) {
    error->clear();
    tensors->clear();

    ggml_context *ctx = nullptr;
    gguf_init_params params{/*no_alloc=*/false, /*ctx=*/&ctx};
    gguf_context *gguf = gguf_init_from_file(path.c_str(), params);
    if (gguf == nullptr || ctx == nullptr) {
        *error = "failed to read GGUF model: " + path;
        return false;
    }

    const int64_t arch_key = gguf_find_key(gguf, "general.architecture");
    if (arch_key < 0) {
        *error = "missing GGUF architecture metadata";
        gguf_free(gguf);
        return false;
    }
    const std::string architecture = gguf_get_val_str(gguf, arch_key);
    if (architecture != kArchitecture) {
        *error = "unsupported GGUF architecture '" + architecture + "'";
        gguf_free(gguf);
        return false;
    }

    const int64_t dim_key = gguf_find_key(gguf, "aliked.descriptor_dim");
    if (dim_key >= 0) {
        *descriptor_dim = static_cast<int32_t>(gguf_get_val_u32(gguf, dim_key));
    }

    const int64_t count = gguf_get_n_tensors(gguf);
    for (int64_t i = 0; i < count; ++i) {
        const char *name = gguf_get_tensor_name(gguf, i);
        ggml_tensor *tensor = ggml_get_tensor(ctx, name);
        std::string local_error;
        std::vector<float> data = CopyTensor(tensor, &local_error);
        if (!local_error.empty()) {
            *error = local_error + " (" + name + ")";
            gguf_free(gguf);
            return false;
        }
        (*tensors)[name] = std::move(data);
    }

    gguf_free(gguf);
    return true;
}

}  // namespace lightglue::aliked_internal
