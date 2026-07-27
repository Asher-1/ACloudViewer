// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "gpu_tensor.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>

#include "ggml_cnn.hpp"

namespace lightglue::aliked_internal {

void GpuTensor::Release() {
    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx != nullptr) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    tensor = nullptr;
}

bool GpuTensor::Allocate(internal::Backend *backend,
                         int32_t width,
                         int32_t height,
                         int32_t channels,
                         GpuTensor *out,
                         std::string *error) {
    if (backend == nullptr || backend->handle == nullptr) {
        if (error) {
            *error = "GPU backend is not initialized";
        }
        return false;
    }

    out->Release();
    out->w = width;
    out->h = height;
    out->c = channels;

    const size_t ctx_size = ggml_tensor_overhead() * 4 + 256 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    out->ctx = ggml_init(params);
    if (out->ctx == nullptr) {
        if (error) {
            *error = "failed to create GPU tensor context";
        }
        return false;
    }

    out->tensor = ggml_new_tensor_4d(out->ctx, GGML_TYPE_F32, width, height,
                                     channels, 1);
    out->buffer = ggml_backend_alloc_ctx_tensors(out->ctx, backend->handle);
    if (out->buffer == nullptr) {
        if (error) {
            *error = "failed to allocate GPU tensor buffer";
        }
        out->Release();
        return false;
    }
    return true;
}

bool GpuTensor::UploadNchw(internal::Backend *backend,
                           const std::vector<float> &nchw,
                           int32_t ic,
                           int32_t ih,
                           int32_t iw,
                           std::string *error) {
    if (tensor == nullptr) {
        if (error) {
            *error = "GPU tensor is not allocated";
        }
        return false;
    }
    std::vector<float> whcn;
    NchwToWhcn(nchw, ic, ih, iw, &whcn);
    ggml_backend_tensor_set(tensor, whcn.data(), 0,
                            whcn.size() * sizeof(float));
    (void)backend;
    return true;
}

bool GpuTensor::DownloadNchw(internal::Backend *backend,
                             std::vector<float> *nchw,
                             int32_t ic,
                             int32_t ih,
                             int32_t iw,
                             std::string *error) const {
    if (tensor == nullptr) {
        if (error) {
            *error = "GPU tensor is not allocated";
        }
        return false;
    }
    std::vector<float> whcn(ElementCount());
    ggml_backend_tensor_get(tensor, whcn.data(), 0,
                            whcn.size() * sizeof(float));
    WhcnToNchw(whcn, ic, ih, iw, nchw);
    (void)backend;
    (void)error;
    return true;
}

}  // namespace lightglue::aliked_internal
