// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "model.h"

#include <ggml.h>

#include <cstdlib>

#include "models/bev_backbone.h"

namespace openpcdet {

bool model::forward_scene(const scene_input& scene,
                          std::vector<detection>& out_dets,
                          const tap_sink& sink) {
    return forward(scene.points, scene.n_points, scene.point_dim, out_dets,
                   sink);
}

bool model::load(const std::string& gguf_path,
                 const std::string& device,
                 int n_threads) {
    release();

    if (!file.open(gguf_path, /*with_data =*/true)) {
        error = file.error;
        return false;
    }
    if (!be.init(device, n_threads)) {
        error = be.error;
        return false;
    }
    if (!realize_weights()) {
        return false;
    }
    return true;
}

void model::release() {
    if (weights_buf) {
        ggml_backend_buffer_free(weights_buf);
        weights_buf = nullptr;
    }
    if (device_ctx) {
        ggml_free(device_ctx);
        device_ctx = nullptr;
    }
    be.release();
    file.close();
    error.clear();
}

bool model::realize_weights() {
    // CUDA-ext fused BN-fold+ReLU (upstream OpenPCDet-GGML ggml op): the
    // conv_bn_relu triple lowers to one fused node on CUDA. Vulkan and CPU
    // keep the portable mul + add + relu triple (byte-identical graphs).
    // Packed direct convolution remains unavailable in this port.
#ifdef AICORE_OPENPCDET_CUDA_EXT
    set_bn_fuse_relu(be.is_cuda());
#endif
    if (be.is_cpu()) {
        // Zero-copy: GGUF was loaded with no_alloc=false, so all tensor data
        // lives in one contiguous ctx buffer. Wrap it as a CPU backend buffer
        // so graphs can reference loader tensors directly as leaves.
        void* base = ggml_get_mem_buffer(file.ctx);
        size_t size = ggml_get_mem_size(file.ctx);
        weights_buf = ggml_backend_cpu_buffer_from_ptr(base, size);
        if (!weights_buf) {
            error = "cpu weight buffer wrap failed";
            return false;
        }
        for (ggml_tensor* t = ggml_get_first_tensor(file.ctx); t;
             t = ggml_get_next_tensor(file.ctx, t)) {
            t->buffer = weights_buf;
        }
        return true;
    }

    // Device path: mirror weights into a no_alloc ctx, allocate on the backend,
    // upload, and repoint the weight pointers at the device tensors.
    size_t n_tensors = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(file.ctx); t;
         t = ggml_get_next_tensor(file.ctx, t)) {
        n_tensors++;
    }
    ggml_init_params dp = {ggml_tensor_overhead() * (n_tensors + 8), nullptr,
                           /*no_alloc =*/true};
    device_ctx = ggml_init(dp);
    if (!device_ctx) {
        error = "device ctx init failed";
        return false;
    }
    for (ggml_tensor* s = ggml_get_first_tensor(file.ctx); s;
         s = ggml_get_next_tensor(file.ctx, s)) {
        ggml_tensor* d =
                ggml_new_tensor(device_ctx, s->type, GGML_MAX_DIMS, s->ne);
        ggml_set_name(d, ggml_get_name(s));
    }
    weights_buf = ggml_backend_alloc_ctx_tensors(device_ctx, be.be);
    if (!weights_buf) {
        error = "device weight alloc failed (out of device memory?)";
        return false;
    }
    for (ggml_tensor* s = ggml_get_first_tensor(file.ctx); s;
         s = ggml_get_next_tensor(file.ctx, s)) {
        ggml_tensor* d = ggml_get_tensor(device_ctx, ggml_get_name(s));
        ggml_backend_tensor_set(d, s->data, 0, ggml_nbytes(s));
    }
    return true;
}

ggml_tensor* model::resolved(const char* name) const {
    // After realize_weights() on a device backend, the weight pointers stored
    // by the subclass point at the GGUF loader ctx — but those tensors have
    // NO buffer. The actual on-device tensor lives in device_ctx under the
    // same name. On CPU, file.ctx tensors are valid directly.
    if (device_ctx) {
        if (ggml_tensor* t = ggml_get_tensor(device_ctx, name)) return t;
    }
    return file.tensor(name);
}

}  // namespace openpcdet
